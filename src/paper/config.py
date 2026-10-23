"""Frozen numerical defaults from main; no parameter search at runtime."""

SOURCE_REVISION = "b0bb808cffa4ef8a1a18e0a60e269a581fbcc01d"

PHM_RAW_SIGNAL_LENGTH = 2560

PARAMS_CPH = {'alpha': 0.0001,
              'tol': 0.1, 
              'n_iter': 50}

PARAMS_RSF = {'n_estimators': 100, 
              'min_samples_split': 2, 
              'min_samples_leaf': 4, 
              'max_depth': 7}

PARAMS_DSM = {
    'k': 3,  # 混合分布数量
    'layers': [32],  # MLP 隐藏层
    'distribution': 'Weibull',  # 'Weibull' 或 'LogNormal'
    'learning_rate': 1e-3,
    'iters': 100,
    'batch_size': 32,
    # The full-development DSM path uses this fixed count without a
    # validation split or early stopping.
    'pretrain_iters': 10000,
}

PARAMS_BNN = {'layers' : [32],
              'regularization_pen': 0.001}

PARAMS_MTLR = {
    'hidden_size': 64,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.00008,
    'num_epochs': 1000,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 50}

PARAMS_COXBOOST = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'loss': 'coxph',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'dropout_rate': 0.0,
    'subsample': 1.0}

PARAMS_R13_OTSURV = {
    'mode': 'r13',
    'loss_fn': 'cox',
    'raw_signal_normalization': 'none',
    'window_length': 512,
    'stride': 512,
    # Same-bearing causal early reference used by the R13 development model.
    'r13_reference_length': 8,
    # Bound the three 64-by-64 Sinkhorn solves held in one autograd chunk.
    'r13_transport_batch_size': 4,
    # Identity-initialized, shared per-channel token projection. Keeping it
    # trainable makes waveform-shape OT part of the joint model.
    'r13_shape_projection_trainable': True,
    'r13_sinkhorn_checkpoint': True,
    'ot_epsilon': 0.5,
    'ot_num_iters': 500,
    'ot_stoperr': 1e-10,
    # Legacy-preserving joint optimization.  Every fitted quantity below is
    # derived from the inner-train bearings only; the final phase unfreezes
    # both branches and remains one end-to-end Cox model.
    'r13_amplitude_init': 'cox_train',
    'r13_amplitude_cox_alpha': 10.0,
    'r13_amplitude_freeze_epochs': 8,
    'r13_shape_projection_freeze_epochs': 8,
    'r13_amplitude_anchor_weight': 0.25,
    'r13_amplitude_aux_cox_weight': 0.25,
    'r13_orthogonalize_trajectory_to_main': True,
    'r13_ot_head_l2_weight': 1.0,
    'r13_ot_risk_orthogonality_weight': 0.1,
    'lr': 0.002,
    'r13_weight_decay': 1e-4,
    # Positive post-Cox risk temperature. It is applied only to the risks
    # passed to Breslow, never inside the exact Cox training graph.
    'r13_risk_temperature': 1.0,
    # This budget is the one validated by the frozen B1--B4 comparison.
    'max_epochs': 30,
    'early_stopping_patience': 12,
    'min_epochs_before_stopping': 12,
}

PARAMS_R13_OTSURV_CONDITION_OVERRIDES = {
    2: {
        'r13_risk_temperature': 0.4,
        # This only batches independent Sinkhorn problems; it is numerically
        # equivalent to the frozen chunk size and matches the formal C3 run.
        'r13_transport_batch_size': 32,
    },
}

PARAMS_R14B_OTSURV = {
    'mode': 'r14b',
    'loss_fn': 'hybrid_cox_aft',
    'raw_signal_normalization': 'none',
    'signal_length': 32768,
    'window_length': 512,
    'stride': 512,
    'r14b_reference_length_short': 4,
    'r14b_reference_length_medium': 8,
    'r14b_reference_length_long': 16,
    'r14b_transport_batch_size': 32,
    'r14b_shape_projection_trainable': True,
    'r14b_sinkhorn_checkpoint': True,
    'ot_epsilon': 0.5,
    'ot_num_iters': 500,
    'ot_stoperr': 1e-10,
    'r14b_state_hidden_dim': 96,
    'r14b_sigma_init': 0.60,
    'r14b_delta_fpt_scale': 0.75,
    'r14b_warmup_epochs': 6,
    'r14b_teacher_epochs': 30,
    'r14b_loss_profile': 'balanced',
    'lr': 0.001,
    'r14b_weight_decay': 1e-4,
    'max_epochs': 40,
    'early_stopping_patience': 10,
    'min_epochs_before_stopping': 8,
    'r14b_aft_horizon': 1024,
}

PARAMS_R14F_OTSURV = {
    **PARAMS_R14B_OTSURV,
    'mode': 'r14f',
    'loss_fn': 'joint_trainable_r13_detached_teacher_r14b_time_expert',
    'r14f_rank_anchor_contract': 'trainable_r13_condition_matched_detached_teacher',
    'r14f_time_profile': 'r14b_balanced_shared_teacher',
    'r14f_fixed_epochs': 30,
    'r14f_teacher_timing': 'post_r13_update_same_epoch_detached',
    # Retain C2 as the direct-construction default.  Public callers must use
    # ``get_hyperparams(condition)`` so the serialized fit contract records
    # the actual XJTU condition.
    'r14f_condition': 1,
}

PARAMS_RAW_CNN_COX = {
    'mode': 'raw-cnn',
    'loss_fn': 'cox',
    'raw_signal_normalization': 'none',
    'signal_length': 32768,
    'window_length': 512,
    'stride': 512,
    'hidden_dim': 64,
    'base_channels': 32,
    'lr': 0.001,
    'weight_decay': 1e-4,
    # This controls micro-batch replay only. The Cox loss is always formed
    # over the complete inner-train or full-development risk set.
    'batch_size': 8,
    'max_epochs': 30,
    'early_stopping_patience': 12,
    'min_epochs_before_stopping': 12,
}

PARAMS_PHM_RAW_CNN_COX = {
    **PARAMS_RAW_CNN_COX,
    'signal_length': PHM_RAW_SIGNAL_LENGTH,
}

PAPER_SEED = 0

TRAIN_EPOCHS = 40

CHECKPOINT_EPOCHS = (5, 10, 15, 20, 25, 30, 35, 40)

REPORT_EPOCHS = {0: 40, 1: 10, 2: 20}

# The later main revision reduced a class-level CNN replay chunk for C3.
# C1/C2 retain the earlier execution constant, absent from parameter CSVs.
TIME_CNN_WINDOW_BATCH_SIZES = {0: 16384, 1: 16384, 2: 8192}

MOVING_AVERAGE_LAG = {0: -1, 1: -3, 2: -5}
