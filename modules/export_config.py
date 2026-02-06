from __future__ import annotations

import numpy as np

import modules.browser_storage as browser_storage
import modules.config_manager as config_manager
from modules.constants import (
    DEFAULT_DIFF_STEP_REL,
    DEFAULT_EA_K_MAX_J_MOL,
    DEFAULT_EA_K_MIN_J_MOL,
    DEFAULT_EA_MAX_J_MOL,
    DEFAULT_EA_MIN_J_MOL,
    DEFAULT_EA_REV_MAX_J_MOL,
    DEFAULT_EA_REV_MIN_J_MOL,
    DEFAULT_K0_ADS_MAX,
    DEFAULT_K0_ADS_MIN,
    DEFAULT_K0_MAX,
    DEFAULT_K0_MIN,
    DEFAULT_K0_REV_MAX,
    DEFAULT_K0_REV_MIN,
    DEFAULT_MAX_NFEV,
    DEFAULT_MAX_NFEV_COARSE,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_N_STARTS,
    DEFAULT_ORDER_MAX,
    DEFAULT_ORDER_MIN,
    DEFAULT_ORDER_REV_MAX,
    DEFAULT_ORDER_REV_MIN,
    DEFAULT_RANDOM_SEED,
)


def _pick_cfg(
    get_cfg,
    key: str,
    default,
    advanced_overrides: dict | None,
):
    if isinstance(advanced_overrides, dict) and key in advanced_overrides:
        return advanced_overrides[key]
    return get_cfg(key, default)


def build_export_config_from_ctx(
    ctx: dict,
    output_mode: str,
    output_species_list: list[str],
    advanced_overrides: dict | None = None,
) -> dict:
    get_cfg = ctx["get_cfg"]

    max_step_fraction = float(
        _pick_cfg(
            get_cfg,
            "max_step_fraction",
            DEFAULT_MAX_STEP_FRACTION,
            advanced_overrides,
        )
    )

    export_cfg = config_manager.collect_config(
        reactor_type=ctx["reactor_type"],
        pfr_flow_model=str(ctx["pfr_flow_model"]),
        kinetic_model=ctx["kinetic_model"],
        solver_method=ctx["solver_method"],
        rtol=float(ctx["rtol"]),
        atol=float(ctx["atol"]),
        max_step_fraction=max_step_fraction,
        species_text=str(ctx["species_text"]),
        n_reactions=int(ctx["n_reactions"]),
        stoich_matrix=np.asarray(ctx["stoich_matrix"], dtype=float),
        order_guess=np.asarray(ctx["order_guess"], dtype=float),
        fit_order_flags_matrix=np.asarray(ctx["fit_order_flags_matrix"], dtype=bool),
        k0_guess=np.asarray(ctx["k0_guess"], dtype=float),
        ea_guess_J_mol=np.asarray(ctx["ea_guess_J_mol"], dtype=float),
        fit_k0_flags=np.asarray(ctx["fit_k0_flags"], dtype=bool),
        fit_ea_flags=np.asarray(ctx["fit_ea_flags"], dtype=bool),
        K0_ads=(
            None
            if ctx["K0_ads"] is None
            else np.asarray(ctx["K0_ads"], dtype=float)
        ),
        Ea_K_J_mol=(
            None
            if ctx["Ea_K_J_mol"] is None
            else np.asarray(ctx["Ea_K_J_mol"], dtype=float)
        ),
        fit_K0_ads_flags=(
            None
            if ctx["fit_K0_ads_flags"] is None
            else np.asarray(ctx["fit_K0_ads_flags"], dtype=bool)
        ),
        fit_Ea_K_flags=(
            None
            if ctx["fit_Ea_K_flags"] is None
            else np.asarray(ctx["fit_Ea_K_flags"], dtype=bool)
        ),
        m_inhibition=(
            None
            if ctx["m_inhibition"] is None
            else np.asarray(ctx["m_inhibition"], dtype=float)
        ),
        fit_m_flags=(
            None
            if ctx["fit_m_flags"] is None
            else np.asarray(ctx["fit_m_flags"], dtype=bool)
        ),
        k0_rev=(
            None
            if ctx["k0_rev"] is None
            else np.asarray(ctx["k0_rev"], dtype=float)
        ),
        ea_rev_J_mol=(
            None
            if ctx["ea_rev_J_mol"] is None
            else np.asarray(ctx["ea_rev_J_mol"], dtype=float)
        ),
        fit_k0_rev_flags=(
            None
            if ctx["fit_k0_rev_flags"] is None
            else np.asarray(ctx["fit_k0_rev_flags"], dtype=bool)
        ),
        fit_ea_rev_flags=(
            None
            if ctx["fit_ea_rev_flags"] is None
            else np.asarray(ctx["fit_ea_rev_flags"], dtype=bool)
        ),
        order_rev=(
            None
            if ctx["order_rev"] is None
            else np.asarray(ctx["order_rev"], dtype=float)
        ),
        fit_order_rev_flags_matrix=(
            None
            if ctx["fit_order_rev_flags_matrix"] is None
            else np.asarray(ctx["fit_order_rev_flags_matrix"], dtype=bool)
        ),
        output_mode=str(output_mode),
        output_species_list=list(output_species_list),
        k0_min=float(_pick_cfg(get_cfg, "k0_min", DEFAULT_K0_MIN, advanced_overrides)),
        k0_max=float(_pick_cfg(get_cfg, "k0_max", DEFAULT_K0_MAX, advanced_overrides)),
        ea_min_J_mol=float(
            _pick_cfg(get_cfg, "ea_min_J_mol", DEFAULT_EA_MIN_J_MOL, advanced_overrides)
        ),
        ea_max_J_mol=float(
            _pick_cfg(get_cfg, "ea_max_J_mol", DEFAULT_EA_MAX_J_MOL, advanced_overrides)
        ),
        order_min=float(
            _pick_cfg(get_cfg, "order_min", DEFAULT_ORDER_MIN, advanced_overrides)
        ),
        order_max=float(
            _pick_cfg(get_cfg, "order_max", DEFAULT_ORDER_MAX, advanced_overrides)
        ),
        K0_ads_min=float(
            _pick_cfg(get_cfg, "K0_ads_min", DEFAULT_K0_ADS_MIN, advanced_overrides)
        ),
        K0_ads_max=float(
            _pick_cfg(get_cfg, "K0_ads_max", DEFAULT_K0_ADS_MAX, advanced_overrides)
        ),
        Ea_K_min=float(
            _pick_cfg(get_cfg, "Ea_K_min", DEFAULT_EA_K_MIN_J_MOL, advanced_overrides)
        ),
        Ea_K_max=float(
            _pick_cfg(get_cfg, "Ea_K_max", DEFAULT_EA_K_MAX_J_MOL, advanced_overrides)
        ),
        k0_rev_min=float(
            _pick_cfg(get_cfg, "k0_rev_min", DEFAULT_K0_REV_MIN, advanced_overrides)
        ),
        k0_rev_max=float(
            _pick_cfg(get_cfg, "k0_rev_max", DEFAULT_K0_REV_MAX, advanced_overrides)
        ),
        ea_rev_min_J_mol=float(
            _pick_cfg(
                get_cfg, "ea_rev_min_J_mol", DEFAULT_EA_REV_MIN_J_MOL, advanced_overrides
            )
        ),
        ea_rev_max_J_mol=float(
            _pick_cfg(
                get_cfg, "ea_rev_max_J_mol", DEFAULT_EA_REV_MAX_J_MOL, advanced_overrides
            )
        ),
        order_rev_min=float(
            _pick_cfg(get_cfg, "order_rev_min", DEFAULT_ORDER_REV_MIN, advanced_overrides)
        ),
        order_rev_max=float(
            _pick_cfg(get_cfg, "order_rev_max", DEFAULT_ORDER_REV_MAX, advanced_overrides)
        ),
        residual_type=str(
            _pick_cfg(get_cfg, "residual_type", "绝对残差", advanced_overrides)
        ),
        diff_step_rel=float(
            _pick_cfg(
                get_cfg, "diff_step_rel", DEFAULT_DIFF_STEP_REL, advanced_overrides
            )
        ),
        max_nfev=int(_pick_cfg(get_cfg, "max_nfev", DEFAULT_MAX_NFEV, advanced_overrides)),
        use_x_scale_jac=bool(
            _pick_cfg(get_cfg, "use_x_scale_jac", True, advanced_overrides)
        ),
        use_multi_start=bool(
            _pick_cfg(get_cfg, "use_multi_start", True, advanced_overrides)
        ),
        n_starts=int(_pick_cfg(get_cfg, "n_starts", DEFAULT_N_STARTS, advanced_overrides)),
        max_nfev_coarse=int(
            _pick_cfg(
                get_cfg, "max_nfev_coarse", DEFAULT_MAX_NFEV_COARSE, advanced_overrides
            )
        ),
        random_seed=int(
            _pick_cfg(get_cfg, "random_seed", DEFAULT_RANDOM_SEED, advanced_overrides)
        ),
    )
    return export_cfg


def persist_export_config(
    export_cfg: dict,
    session_id: str | None,
) -> tuple[bool, str]:
    is_valid_cfg, _ = config_manager.validate_config(export_cfg)
    if not is_valid_cfg:
        return True, ""

    ok, message = config_manager.auto_save_config(export_cfg, session_id)
    if not ok:
        return False, message
    browser_storage.save_config_to_browser(export_cfg)
    return True, "OK"


def render_export_config_button(
    placeholder,
    export_cfg: dict,
    button_key: str,
) -> None:
    export_config_bytes = config_manager.export_config_to_json(export_cfg).encode(
        "utf-8"
    )
    placeholder.download_button(
        "📥 导出当前配置 (JSON)",
        export_config_bytes,
        file_name="kinetics_config.json",
        mime="application/json",
        use_container_width=True,
        key=button_key,
    )
