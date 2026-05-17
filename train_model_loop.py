# ============================================================
#  LSTM + Evidential Deep Learning (Normal-Gamma) Training Script
#  - Satellite-wise sequential split (train/val/test per satellite)
#  - Optional multi-level storm rebalancing (currently OFF by CONFIG)
#  - Train multiple seeds (10..19)
#  - Evaluate external tests using saved best weights
#  - Save per-seed outputs + a summary MAT file
# ============================================================

import os
import glob
import random
import numpy as np
import h5py
import scipy.io as io
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, norm

import evidential_deep_learning as edl


# ===================== Repro / Runtime Flags =====================
# - Full determinism on GPU is tricky; these flags help but are not perfect across drivers/CUDA/cuDNN.
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Enable dynamic GPU memory growth (avoid pre-alloc all VRAM)
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ===================== CONFIG =====================
CONFIG = {
    # ------------------ Dataset split ------------------
    "SAT_LENGTHS":      [478683, 356913, 176474, 183840],  # CHAMP, GRACE, SWARM-C, GOCE
    "SAT_VAL_LENGTH":   20000,
    "SAT_TEST_LENGTH":  5000,

    # ------------------ Training hyperparams ------------------
    "BATCH_SIZE": 256,
    "EPOCHS": 500,
    "LR": 5e-5,
    "CLIPNORM": 1.0,

    # EDL loss coefficient
    "EDL_LOSS_COEFF": 1e-5,

    # callbacks
    "PATIENCE_ES": 30,
    "PATIENCE_RLRP": 8,
    "RLRP_FACTOR": 0.5,
    "MIN_LR": 1e-6,

    # tf.data pipeline
    "SHUFFLE_BUFFER": 131072,
    "PREFETCH_BUFSIZE": tf.data.AUTOTUNE,

    # IO
    "Folder_DIR": "Output/Model/Model1",

    # ------------------ Model architecture ------------------
    "LSTM_UNITS_HIST": 192,   # BiLSTM units (per direction); output dim is 2*units
    "LSTM_UNITS_FUT":  128,   # LSTM units for future driver branch
    "DROPOUT_LSTM": 0.20,

    "DENSE_HIDDEN_1": 256,
    "DENSE_HIDDEN_2": 192,
    "DROPOUT_FC1": 0.15,
    "DROPOUT_FC2": 0.15,
    "HEAD_DROPOUT": 0.10,

    # ------------------ Optional auxiliary mu-MSE term (disabled by default) ------------------
    "MU_MSE_W0": 0.0,        # 0.0 => pure evidential loss
    "MU_MSE_WARMUP": 1,      # epochs to decay from w0 -> 0 (linear)

    # ------------------ Storm levels (Dst thresholds) ------------------
    "DST_FEATURE_INDEX": 3,   # feature index inside x_hist last dimension

    "STORM_L1_THRESH": -50.0,   # mild/moderate
    "STORM_L2_THRESH": -80.0,   # strong
    "STORM_L3_THRESH": -150.0,  # severe

    # repeat multipliers for rebalancing (currently OFF)
    "REP_QUIET": 1,
    "REP_STORM_L1": 2,
    "REP_STORM_L2": 4,
    "REP_STORM_L3": 8,

    "USE_STORM_REBALANCE": False,
}


# ============================================================
#  Model builder
# ============================================================
def build_model(m, Hh, n, Hf):
    """
    Old stable architecture (no cross-attention):
      x_hist (N, m, Hh) -> BiLSTM -> h_vec
      x_fut  (N, n, Hf) ->  LSTM  -> f_vec
      concat -> BN -> Dense -> Dense -> DenseNormalGamma(n)

    Inputs:
      m  : history length (timesteps)
      Hh : history feature dimension
      n  : forecast horizon length
      Hf : future-driver feature dimension

    Output:
      DenseNormalGamma(n): concatenated [mu, v, alpha, beta] along last axis
    """

    lstm_units_hist = int(CONFIG["LSTM_UNITS_HIST"])
    lstm_units_fut  = int(CONFIG["LSTM_UNITS_FUT"])
    lstm_dp         = float(CONFIG["DROPOUT_LSTM"])

    fc1_units = int(CONFIG["DENSE_HIDDEN_1"])
    fc2_units = int(CONFIG["DENSE_HIDDEN_2"])
    fc1_dp    = float(CONFIG["DROPOUT_FC1"])
    fc2_dp    = float(CONFIG["DROPOUT_FC2"])
    head_dp   = float(CONFIG["HEAD_DROPOUT"])

    l2_reg = tf.keras.regularizers.l2(1e-4)

    # Inputs
    inp_hist = L.Input(shape=(m, Hh), name="x_hist")
    inp_fut  = L.Input(shape=(n, Hf), name="x_fut")

    # History encoder
    h_vec = L.Bidirectional(
        L.LSTM(
            lstm_units_hist,
            return_sequences=False,
            dropout=lstm_dp,
            recurrent_dropout=0.0,
            name="lstm_hist_core",
        ),
        name="bilstm_hist",
    )(inp_hist)

    # Future driver encoder
    f_vec = L.LSTM(
        lstm_units_fut,
        return_sequences=False,
        dropout=lstm_dp,
        recurrent_dropout=0.0,
        name="lstm_fut_core",
    )(inp_fut)

    # Merge
    x = L.Concatenate(name="concat_hist_fut")([h_vec, f_vec])
    x = L.BatchNormalization(name="concat_bn")(x)

    # Dense head
    x = L.Dense(fc1_units, activation="elu",
                kernel_initializer="he_normal",
                kernel_regularizer=l2_reg, name="fc1")(x)
    x = L.Dropout(fc1_dp, name="fc1_dp")(x)

    x = L.Dense(fc2_units, activation="elu",
                kernel_initializer="he_normal",
                kernel_regularizer=l2_reg, name="fc2")(x)
    x = L.Dropout(fc2_dp, name="fc2_dp")(x)

    if head_dp > 0:
        x = L.Dropout(head_dp, name="head_dp")(x)

    edl_out = edl.layers.DenseNormalGamma(n)(x)

    return Model(inputs=[inp_hist, inp_fut], outputs=edl_out)


# ============================================================
#  Data IO utilities
# ============================================================
def _read_h5_dataset(f, key_candidates):
    """
    Robustly read a dataset from an HDF5 file, trying candidate keys.
    Returns ndarray or raises KeyError if none found.
    """
    for k in key_candidates:
        if k in f:
            return f[k][()]
    raise KeyError(f"None of the keys exist in file: {key_candidates}")


def load_data(data_dir):
    """
    Expected files under data_dir:
      - ydata.mat
      - x_fut_data.mat
      - x_hist_data_with_AE.mat  (preferred)
        OR x_hist_data.mat        (fallback)

    Expected dataset names (your files vary):
      y:  ydata
      xf: x_fut
      xh: x_hist_with_AE  OR x_hist
    """
    y_path  = os.path.join(data_dir, "ydata.mat")
    xf_path = os.path.join(data_dir, "x_fut_data.mat")

    # hist may come in two possible filenames
    xh_path_with_ae = os.path.join(data_dir, "x_hist_data_with_AE.mat")
    xh_path_plain   = os.path.join(data_dir, "x_hist_data.mat")

    if os.path.exists(xh_path_with_ae):
        xh_path = xh_path_with_ae
    elif os.path.exists(xh_path_plain):
        xh_path = xh_path_plain
    else:
        raise FileNotFoundError(f"No x_hist mat found in {data_dir}")

    for p in [y_path, xf_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with h5py.File(y_path, "r") as fy, h5py.File(xh_path, "r") as fxh, h5py.File(xf_path, "r") as fxf:
        y  = _read_h5_dataset(fy,  ["ydata"]).transpose().astype(np.float32)

        # x_hist keys differ across your datasets
        xh_raw = _read_h5_dataset(fxh, ["x_hist_with_AE", "x_hist"]).transpose().astype(np.float32)

        xf = _read_h5_dataset(fxf, ["x_fut"]).transpose().astype(np.float32)

    return xh_raw, xf, y


def split_sat(N):
    """
    Satellite-wise sequential split:
      per satellite: [train | val | test] in time order
    """
    SAT_LENGTHS     = list(map(int, CONFIG["SAT_LENGTHS"]))
    SAT_VAL_LENGTH  = int(CONFIG["SAT_VAL_LENGTH"])
    SAT_TEST_LENGTH = int(CONFIG["SAT_TEST_LENGTH"])

    total = sum(SAT_LENGTHS)
    if int(N) != total:
        raise ValueError(f"N ({N}) != sum(SAT_LENGTHS) ({total}).")

    tr_list, va_list, te_list = [], [], []
    offset = 0

    for i, Lsat in enumerate(SAT_LENGTHS):
        if SAT_VAL_LENGTH + SAT_TEST_LENGTH > Lsat:
            raise ValueError(
                f"Satellite {i}: length {Lsat} < VAL({SAT_VAL_LENGTH}) + TEST({SAT_TEST_LENGTH})."
            )

        n_train = Lsat - SAT_VAL_LENGTH - SAT_TEST_LENGTH

        # train: head
        tr_rel = np.arange(0, n_train, dtype=np.int64)
        # val: middle
        va_rel = np.arange(n_train, n_train + SAT_VAL_LENGTH, dtype=np.int64)
        # test: tail
        te_rel = np.arange(n_train + SAT_VAL_LENGTH, Lsat, dtype=np.int64)

        tr_list.append(tr_rel + offset)
        va_list.append(va_rel + offset)
        te_list.append(te_rel + offset)

        offset += Lsat

    return np.concatenate(tr_list), np.concatenate(va_list), np.concatenate(te_list)


# ============================================================
#  Storm-level labeling and rebalanced tf.data pipeline
# ============================================================
def compute_storm_level_from_hist(xh, dst_feature_index):
    """
    Determine storm severity per sample using min(Dst) over the history window.

    Returns:
      level: int8 array shape (N,)
        0 = quiet
        1 = mild/moderate
        2 = strong
        3 = severe/extreme
    """
    dst_hist = xh[..., dst_feature_index]      # (N, m)
    min_dst  = np.min(dst_hist, axis=1)        # (N,)

    L1 = CONFIG["STORM_L1_THRESH"]
    L2 = CONFIG["STORM_L2_THRESH"]
    L3 = CONFIG["STORM_L3_THRESH"]

    level = np.zeros_like(min_dst, dtype=np.int8)
    level[(min_dst <= L1) & (min_dst > L2)] = 1
    level[(min_dst <= L2) & (min_dst > L3)] = 2
    level[(min_dst <= L3)] = 3

    print("[Storm Level Distribution]")
    N = len(level)
    for lv in range(4):
        cnt = int(np.sum(level == lv))
        print(f"  level {lv}: {cnt} samples ({cnt / max(1, N):.2%})")

    return level


def make_datasets_multilevel_storm(
    xh_tr, xf_tr, y_tr,
    xh_va, xf_va, y_va,
    xh_te, xf_te, y_te,
    storm_level_tr,
    seed, batch_size,
    shuffle_buffer, prefetch_bufsize,
    use_rebalance
):
    """
    Build tf.data datasets.
    If use_rebalance=True, oversample storm levels via dataset repeat.
    """

    def base_ds(xh_, xf_, y_, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(((xh_, xf_), y_))
        if shuffle:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(prefetch_bufsize)

    if not use_rebalance:
        ds_tr = base_ds(xh_tr, xf_tr, y_tr, shuffle=True)
    else:
        idx0 = np.where(storm_level_tr == 0)[0]
        idx1 = np.where(storm_level_tr == 1)[0]
        idx2 = np.where(storm_level_tr == 2)[0]
        idx3 = np.where(storm_level_tr == 3)[0]

        def make_rep_ds(idx, rep, tag):
            if idx.size == 0 or rep <= 0:
                print(f"[Storm Rebalance] {tag}: 0 samples or rep<=0, skipped.")
                return None
            print(f"[Storm Rebalance] {tag}: {idx.size} samples, repeat={rep}")
            ds = tf.data.Dataset.from_tensor_slices(((xh_tr[idx], xf_tr[idx]), y_tr[idx]))
            ds = ds.shuffle(10000, seed=seed, reshuffle_each_iteration=True).repeat(rep)
            return ds

        ds_list = [
            make_rep_ds(idx0, CONFIG["REP_QUIET"],    "L0 quiet"),
            make_rep_ds(idx1, CONFIG["REP_STORM_L1"], "L1 mild"),
            make_rep_ds(idx2, CONFIG["REP_STORM_L2"], "L2 strong"),
            make_rep_ds(idx3, CONFIG["REP_STORM_L3"], "L3 severe"),
        ]
        ds_list = [d for d in ds_list if d is not None]

        if len(ds_list) == 0:
            print("[Storm Rebalance] No data to rebalance; fallback to normal shuffle.")
            ds_tr = base_ds(xh_tr, xf_tr, y_tr, shuffle=True)
        else:
            ds_tr = ds_list[0]
            for d in ds_list[1:]:
                ds_tr = ds_tr.concatenate(d)
            ds_tr = ds_tr.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
            ds_tr = ds_tr.batch(batch_size).prefetch(prefetch_bufsize)

    ds_va = base_ds(xh_va, xf_va, y_va, shuffle=False)
    ds_te = base_ds(xh_te, xf_te, y_te, shuffle=False)
    return ds_tr, ds_va, ds_te


# ============================================================
#  Metrics + plotting (log10 domain inputs)
# ============================================================
def compute_metrics_full(y_true_log, mu_log, var_log,
                         valid_min_log=-20, valid_max_log=0):
    """
    Your metric convention:
      - y_true_log, mu_log are log10(density)
      - convert to physical domain via 10^x for R and RMSE
      - coverage and MACE are computed in log space using sigma_log = sqrt(var_log)
      - Relative error uses 10^(mu_log - y_true_log)

    Returns a dict with:
      R, RMSE, Coverage_2sigma, MACE, Relative_Error, N_kept
    """
    y_log = np.asarray(y_true_log, dtype=np.float64).reshape(-1)
    m_log = np.asarray(mu_log,     dtype=np.float64).reshape(-1)
    s_log = np.sqrt(np.maximum(np.asarray(var_log, dtype=np.float64), 0.0)).reshape(-1)

    mask = (
        np.isfinite(y_log) & np.isfinite(m_log) & np.isfinite(s_log) &
        (y_log >= valid_min_log) & (y_log <= valid_max_log)
    )

    y_log, m_log, s_log = y_log[mask], m_log[mask], s_log[mask]
    n = int(y_log.size)

    if n == 0:
        return dict(R=np.nan, RMSE=np.nan, Coverage_2sigma=np.nan,
                    MACE=np.nan, Relative_Error=np.nan, N_kept=0)

    y_phy = np.power(10.0, y_log)
    m_phy = np.power(10.0, m_log)

    if n < 2 or np.all(y_phy == y_phy[0]) or np.all(m_phy == m_phy[0]):
        R = np.nan
    else:
        R = float(pearsonr(y_phy, m_phy)[0])

    RMSE = float(np.sqrt(np.mean((y_phy - m_phy) ** 2)))
    coverage_2s = float(np.mean(np.abs(y_log - m_log) <= 2.0 * s_log))

    # MACE calibration error across confidence levels
    CL = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
                   0.80, 0.90, 0.95, 0.96, 0.99], dtype=np.float64)
    diffs = []
    for cl in CL:
        z = norm.ppf(0.5 * (1.0 + cl))
        emp = np.mean(np.abs(y_log - m_log) <= z * s_log)
        diffs.append(abs(emp - cl))
    MACE = float(np.mean(diffs))

    rel = float(np.mean(np.abs(1.0 - np.power(10.0, m_log - y_log))))

    return dict(R=R, RMSE=RMSE, Coverage_2sigma=coverage_2s,
                MACE=MACE, Relative_Error=rel, N_kept=n)


def plot_full_series(y_true_log, y_pred_log, sigma_log, save_path, title):
    """
    Flattened visualization over all samples and horizons.
    """
    yt = np.asarray(y_true_log).reshape(-1)
    yp = np.asarray(y_pred_log).reshape(-1)
    sg = np.asarray(sigma_log).reshape(-1)

    x = np.arange(len(yt))
    ub, lb = yp + 2 * sg, yp - 2 * sg

    plt.figure(figsize=(14, 4), dpi=140)
    pred_color = plt.get_cmap("tab10")(0)
    plt.fill_between(x, lb, ub, color=pred_color, alpha=0.3, label="±2σ")
    plt.plot(x, yt, color="black", linewidth=1.5, label="Truth")
    plt.plot(x, yp, color=pred_color, linestyle="--", linewidth=1.7, label="Pred")
    plt.title(title)
    plt.xlabel("Index (flattened over horizons)")
    plt.ylabel("log10(density)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# ============================================================
#  Training routine (single seed)
# ============================================================
def train_one_seed(train_dir, seed):
    """
    Train one seed:
      - load training dataset
      - satellite-wise split
      - fit scalers on train only
      - train model
      - evaluate internal test split
      - save weights + diagnostics
    """
    os.makedirs("Parameter", exist_ok=True)
    os.makedirs("Model", exist_ok=True)
    os.makedirs("Image", exist_ok=True)

    # ---- Load raw data ----
    xh, xf, y = load_data(train_dir)
    N, m, Hh = xh.shape
    _, n, Hf = xf.shape

    # ---- Storm labeling from raw x_hist (before normalization) ----
    storm_level_full = compute_storm_level_from_hist(xh, CONFIG["DST_FEATURE_INDEX"])

    # ---- Satellite-wise sequential split ----
    idx_tr, idx_va, idx_te = split_sat(N)

    # ---- Standardize inputs (fit on TRAIN only) ----
    xh_scaler = StandardScaler()
    xf_scaler = StandardScaler()

    xh_scaler.fit(xh[idx_tr].reshape(-1, Hh))
    xf_scaler.fit(xf[idx_tr].reshape(-1, Hf))

    xh_norm = xh_scaler.transform(xh.reshape(-1, Hh)).reshape(N, m, Hh)
    xf_norm = xf_scaler.transform(xf.reshape(-1, Hf)).reshape(N, n, Hf)

    # ---- Normalize y (fit on TRAIN only) ----
    y_mean = np.mean(y[idx_tr], axis=0).astype(np.float64)
    y_std  = np.std(y[idx_tr], axis=0, ddof=0).astype(np.float64)
    y_std  = np.maximum(y_std, 1e-12)  # avoid div-by-zero
    y_norm = (y - y_mean[None, :]) / y_std[None, :]

    # Save normalization parameters ONCE (same split across seeds -> same params)
    nor_path = "Parameter/Nor_parameters.mat"
    if not os.path.exists(nor_path):
        io.savemat(nor_path, {
            "xh_mean": xh_scaler.mean_,
            "xh_std":  xh_scaler.scale_,
            "xf_mean": xf_scaler.mean_,
            "xf_std":  xf_scaler.scale_,
            "y_mean":  y_mean,
            "y_std":   y_std,
        })

    # ---- Create split subsets (normalized) ----
    xh_tr, xf_tr, y_tr = xh_norm[idx_tr], xf_norm[idx_tr], y_norm[idx_tr]
    xh_va, xf_va, y_va = xh_norm[idx_va], xf_norm[idx_va], y_norm[idx_va]
    xh_te, xf_te, y_te = xh_norm[idx_te], xf_norm[idx_te], y_norm[idx_te]

    storm_level_tr = storm_level_full[idx_tr]

    # ---- tf.data pipelines ----
    ds_tr, ds_va, ds_te = make_datasets_multilevel_storm(
        xh_tr, xf_tr, y_tr,
        xh_va, xf_va, y_va,
        xh_te, xf_te, y_te,
        storm_level_tr=storm_level_tr,
        seed=seed,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle_buffer=CONFIG["SHUFFLE_BUFFER"],
        prefetch_bufsize=CONFIG["PREFETCH_BUFSIZE"],
        use_rebalance=CONFIG["USE_STORM_REBALANCE"],
    )

    # ---- Build + compile model ----
    tf.keras.backend.clear_session()
    model = build_model(m, Hh, n, Hf)

    opt = tf.keras.optimizers.Adam(learning_rate=CONFIG["LR"], clipnorm=CONFIG["CLIPNORM"])

    def evidential_loss(y_true, pred):
        return edl.losses.EvidentialRegression(y_true, pred, coeff=CONFIG["EDL_LOSS_COEFF"])

    model.compile(optimizer=opt, loss=evidential_loss)

    # ---- Optional auxiliary mu-MSE term with linear decay ----
    class MuMSEDecay(tf.keras.callbacks.Callback):
        def __init__(self, w0, warmup):
            super().__init__()
            self.w0 = float(w0)
            self.warmup = int(warmup)

        def on_epoch_begin(self, epoch, logs=None):
            # linear decay from w0 to 0
            q = max(0.0, 1.0 - (epoch + 1) / max(1, self.warmup))
            w_t = self.w0 * q

            def combined(y_true, pred):
                mu, _, _, _ = tf.split(pred, 4, axis=-1)
                edl_term = edl.losses.EvidentialRegression(
                    y_true, pred, coeff=CONFIG["EDL_LOSS_COEFF"]
                )
                if w_t > 0.0:
                    return edl_term + w_t * tf.reduce_mean(tf.square(y_true - mu))
                return edl_term

            self.model.loss = combined

    # ---- Callbacks ----
    ckpt_path = f"Model/best_Seed_{seed}.weights.h5"
    cbs = [
        MuMSEDecay(CONFIG["MU_MSE_W0"], CONFIG["MU_MSE_WARMUP"]),
        EarlyStopping(monitor="val_loss",
                      patience=max(CONFIG["PATIENCE_ES"], 20),
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss",
                          patience=CONFIG["PATIENCE_RLRP"],
                          factor=CONFIG["RLRP_FACTOR"],
                          min_lr=CONFIG["MIN_LR"]),
        ModelCheckpoint(ckpt_path, monitor="val_loss", mode="min",
                        save_best_only=True, save_weights_only=True, verbose=0),
        CSVLogger(f"Model/train_log_Seed_{seed}.csv", append=False),
    ]

    # ---- Train ----
    history = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=CONFIG["EPOCHS"],
        callbacks=cbs,
        verbose=1,
    )

    # ---- Save final weights (in addition to best) ----
    model.save_weights(f"Model/model_Seed_{seed}.weights.h5")
    try:
        model.save("Model/full_model.keras")
    except Exception:
        pass

    # ---- Plot loss curves ----
    plt.figure()
    plt.plot(history.history.get("loss", []), label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.title("EDL Training Loss")
    plt.savefig(f"Image/lossplot_EDL_Seed_{seed}.png", dpi=150)
    plt.close()

    # ---- Internal test evaluation ----
    pred = model.predict(ds_te, verbose=0)
    mu_raw, v_raw, alpha_raw, beta_raw = tf.split(pred, 4, axis=-1)

    mu = mu_raw.numpy()
    v = v_raw.numpy()
    alpha = alpha_raw.numpy()
    beta = beta_raw.numpy()

    # Stabilize NG parameters
    v     = np.clip(v, 1e-6, 1e6)
    alpha = np.clip(alpha, 1.0 + 1e-6, 1e6)
    beta  = np.clip(beta, 1e-6, 1e6)

    # Total predictive variance = epistemic + aleatoric (log space)
    epistemic_var = beta / (v * (alpha - 1.0))
    aleatoric_var = beta / (alpha - 1.0)
    pred_var      = np.minimum(epistemic_var + aleatoric_var, 1e6)
    pred_sigma    = np.sqrt(pred_var)

    # Save raw NG outputs (normalized space)
    io.savemat(f"Parameter/mu_Seed_{seed}.mat",    {"mu": mu})
    io.savemat(f"Parameter/v_Seed_{seed}.mat",     {"v": v})
    io.savemat(f"Parameter/alpha_Seed_{seed}.mat", {"alpha": alpha})
    io.savemat(f"Parameter/beta_Seed_{seed}.mat",  {"beta": beta})
    io.savemat(f"Parameter/var_Seed_{seed}.mat",   {"var": pred_var})
    io.savemat(f"Parameter/sigma_Seed_{seed}.mat", {"sigma": pred_sigma})

    # Denormalize to log10 space
    mu_den     = mu * y_std[None, :] + y_mean[None, :]
    var_den    = pred_var * (y_std[None, :] ** 2)
    y_true_den = y_te * y_std[None, :] + y_mean[None, :]

    metrics_all = compute_metrics_full(y_true_den, mu_den, var_den)

    print("\n[Internal Test Metrics]")
    for k, v_ in metrics_all.items():
        print(f"{k}: {v_}")

    plot_full_series(
        y_true_log=y_true_den,
        y_pred_log=mu_den,
        sigma_log=np.sqrt(var_den),
        save_path=f"Image/forecast_vs_truth_log10_Seed_{seed}.png",
        title="Internal Test: Forecast vs Truth (log10, ±2σ)",
    )

    return model, dict(y_mean=y_mean, y_std=y_std, m=m, Hh=Hh, n=n, Hf=Hf)


# ============================================================
#  External evaluation
# ============================================================
def evaluate_external_tests(test_dir, model_dir, seed):
    """
    Load saved weights for a given seed, run prediction on each external test folder,
    save mu/var/y_true, and compute metrics.

    Expects normalization params at Parameter/Nor_parameters.mat.
    """
    tf.keras.backend.clear_session()

    nor_path = os.path.join("Parameter", "Nor_parameters.mat")
    if not os.path.exists(nor_path):
        raise FileNotFoundError(f"{nor_path} not found.")

    params = io.loadmat(nor_path)
    xh_mean = params["xh_mean"].ravel()
    xh_std  = params["xh_std"].ravel()
    xf_mean = params["xf_mean"].ravel()
    xf_std  = params["xf_std"].ravel()
    y_mean  = params["y_mean"].ravel()
    y_std   = params["y_std"].ravel()

    # Discover tests
    test_folders = sorted(glob.glob(os.path.join(test_dir, "Test*")))
    if not test_folders:
        print(f"[Warning] No external test folders found in {test_dir}")
        return {"Seed": int(seed), "Results": {}}

    # Use first test to infer shapes (more robust than assuming TRAIN_DIR layout)
    xh0, xf0, _ = load_data(test_folders[0])
    _, m, Hh = xh0.shape
    _, n, Hf = xf0.shape

    model = build_model(m, Hh, n, Hf)

    best_w = os.path.join(model_dir, f"best_Seed_{seed}.weights.h5")
    fallback_w = os.path.join(model_dir, f"model_Seed_{seed}.weights.h5")

    if os.path.exists(best_w):
        model.load_weights(best_w)
        print(f"[Info] Loaded weights: {best_w}")
    elif os.path.exists(fallback_w):
        model.load_weights(fallback_w)
        print(f"[Info] Loaded weights: {fallback_w}")
    else:
        raise FileNotFoundError(f"No weights found for seed={seed} under {model_dir}")

    results = {"Seed": int(seed), "Results": {}}

    print("\n================ External Test Evaluation ================")
    for tdir in test_folders:
        tname = os.path.basename(tdir)
        print(f"\n--- Evaluating {tname} ---")

        try:
            xh_te, xf_te, y_te = load_data(tdir)
        except Exception as e:
            print(f"[Skip] {tname}: cannot load ({e})")
            continue

        # shape safety
        if xh_te.shape[1:] != (m, Hh) or xf_te.shape[1:] != (n, Hf):
            print(f"[Skip] {tname}: shape mismatch with model")
            continue

        # Normalize with saved train statistics
        xh_te_norm = (xh_te - xh_mean) / xh_std
        xf_te_norm = (xf_te - xf_mean) / xf_std

        ds_te = tf.data.Dataset.from_tensor_slices(((xh_te_norm, xf_te_norm), y_te)) \
            .batch(CONFIG["BATCH_SIZE"])

        pred = model.predict(ds_te, verbose=0)
        mu_raw, v_raw, alpha_raw, beta_raw = tf.split(pred, 4, axis=-1)
        mu = mu_raw.numpy()
        v = v_raw.numpy()
        alpha = alpha_raw.numpy()
        beta = beta_raw.numpy()

        v     = np.clip(v, 1e-6, 1e6)
        alpha = np.clip(alpha, 1.0 + 1e-6, 1e6)
        beta  = np.clip(beta, 1e-6, 1e6)

        epistemic_var = beta / (v * (alpha - 1.0))
        aleatoric_var = beta / (alpha - 1.0)
        pred_var      = np.minimum(epistemic_var + aleatoric_var, 1e6)

        # Denormalize to log10 domain
        mu_den  = mu * y_std[None, :] + y_mean[None, :]
        var_den = pred_var * (y_std[None, :] ** 2)

        # Save per-test outputs
        save_dir = os.path.join("External_Test", f"Seed_{seed}", tname)
        os.makedirs(save_dir, exist_ok=True)
        io.savemat(os.path.join(save_dir, "mu.mat"),     {"mu": mu_den})
        io.savemat(os.path.join(save_dir, "var.mat"),    {"variance": var_den})
        io.savemat(os.path.join(save_dir, "y_true.mat"), {"y_true": y_te})

        metrics = compute_metrics_full(y_te, mu_den, var_den)
        results["Results"][tname] = metrics

        print(f"R={metrics['R']:.4f}, RMSE={metrics['RMSE']:.4e}, "
              f"Coverage={metrics['Coverage_2sigma']:.4f}, "
              f"MACE={metrics['MACE']:.4f}, RE={metrics['Relative_Error']:.4f}")

        plot_full_series(
            y_true_log=y_te,
            y_pred_log=mu_den,
            sigma_log=np.sqrt(var_den),
            save_path=os.path.join("Image", f"{tname}_Seed_{seed}_forecast_vs_truth_log10.png"),
            title=f"{tname}: Forecast vs Truth (log10, ±2σ)",
        )

    print("\n================ Finished External Tests =================\n")
    return results


# ============================================================
#  Main (multi-seed loop)
# ============================================================
if __name__ == "__main__":
    folder_dir = CONFIG["Folder_DIR"]
    train_dir = os.path.join(folder_dir, "Train")
    test_dir  = os.path.join(folder_dir, "Test")

    # Discover test names for consistent summary array
    test_folders = sorted(glob.glob(os.path.join(test_dir, "Test*")))
    test_names = [os.path.basename(p) for p in test_folders]
    metric_names = ["R", "RMSE", "Coverage_2sigma", "MACE", "Relative_Error", "N_kept"]

    seeds = list(range(10, 20))
    metrics_arr = np.full((len(seeds), len(test_names), len(metric_names)), np.nan, dtype=np.float64)
    all_results = []

    for i, seed in enumerate(seeds):
        print("\n" + "=" * 90)
        print(f"==================== Training + External Tests (SEED={seed}) ====================")
        print("=" * 90)

        # Reproducibility per seed
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Train
        _model, _shape_info = train_one_seed(train_dir, seed)

        # External tests
        res = evaluate_external_tests(test_dir=test_dir, model_dir="Model", seed=seed)
        all_results.append(res)

        # Fill summary cube
        for j, tname in enumerate(test_names):
            mm = res.get("Results", {}).get(tname, None)
            if mm is None:
                continue
            for k, mn in enumerate(metric_names):
                try:
                    metrics_arr[i, j, k] = float(mm.get(mn, np.nan))
                except Exception:
                    metrics_arr[i, j, k] = np.nan

    # Save summary as MAT
    os.makedirs("Parameter", exist_ok=True)
    io.savemat(
        "Parameter/ExternalTest_Summary_Seed10_19.mat",
        {
            "seeds": np.array(seeds, dtype=np.int32),
            "test_names": np.array(test_names, dtype=object),
            "metric_names": np.array(metric_names, dtype=object),
            "metrics": metrics_arr,
            "results_struct": all_results,
        }
    )
    print("\n[Saved] Parameter/ExternalTest_Summary_Seed10_19.mat\n")
