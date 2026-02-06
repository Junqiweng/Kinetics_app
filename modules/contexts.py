from __future__ import annotations

from typing import Any, Callable, TypedDict


class BootstrapState(TypedDict):
    session_id: str | None
    main_tab_labels: list[str]
    set_active_main_tab: Callable[[str], None]
    restore_active_main_tab: Callable[[], None]
    request_start_fitting: Callable[[], None]
    request_stop_fitting: Callable[[], None]
    get_cfg: Callable[[str, Any], Any]
    show_help_dialog: Callable[[], None]


class SidebarState(TypedDict):
    export_config_placeholder: Any
    reactor_type: str
    pfr_flow_model: str
    kinetic_model: str
    solver_method: str
    rtol: float
    atol: float


class ModelState(TypedDict, total=False):
    species_text: str
    n_reactions: int
    species_names: list[str]
    stoich_matrix: Any
    k0_guess: Any
    ea_guess_J_mol: Any
    fit_k0_flags: Any
    fit_ea_flags: Any
    order_guess: Any
    fit_order_flags_matrix: Any
    K0_ads: Any
    Ea_K_J_mol: Any
    fit_K0_ads_flags: Any
    fit_Ea_K_flags: Any
    m_inhibition: Any
    fit_m_flags: Any
    k0_rev: Any
    ea_rev_J_mol: Any
    fit_k0_rev_flags: Any
    fit_ea_rev_flags: Any
    order_rev: Any
    fit_order_rev_flags_matrix: Any


class DataState(TypedDict, total=False):
    data_df: Any
    output_mode: str
    output_species_list: list[str]


class BaseContext(TypedDict, total=False):
    session_id: str | None
    main_tab_labels: list[str]
    set_active_main_tab: Callable[[str], None]
    restore_active_main_tab: Callable[[], None]
    request_start_fitting: Callable[[], None]
    request_stop_fitting: Callable[[], None]
    get_cfg: Callable[[str, Any], Any]
    show_help_dialog: Callable[[], None]
    export_config_placeholder: Any
    reactor_type: str
    pfr_flow_model: str
    kinetic_model: str
    solver_method: str
    rtol: float
    atol: float


class FitContext(BaseContext, total=False):
    species_text: str
    n_reactions: int
    species_names: list[str]
    stoich_matrix: Any
    order_guess: Any
    fit_order_flags_matrix: Any
    k0_guess: Any
    ea_guess_J_mol: Any
    fit_k0_flags: Any
    fit_ea_flags: Any
    K0_ads: Any
    Ea_K_J_mol: Any
    fit_K0_ads_flags: Any
    fit_Ea_K_flags: Any
    m_inhibition: Any
    fit_m_flags: Any
    k0_rev: Any
    ea_rev_J_mol: Any
    fit_k0_rev_flags: Any
    fit_ea_rev_flags: Any
    order_rev: Any
    fit_order_rev_flags_matrix: Any
    data_df: Any
    output_mode: str
    output_species_list: list[str]


def build_sidebar_context(bootstrap_state: BootstrapState) -> dict:
    return {
        "get_cfg": bootstrap_state["get_cfg"],
        "show_help_dialog": bootstrap_state["show_help_dialog"],
    }


def build_base_context(
    bootstrap_state: BootstrapState,
    sidebar_state: SidebarState,
) -> dict:
    return {**bootstrap_state, **sidebar_state}


def build_data_context(base_context: BaseContext, model_state: ModelState) -> dict:
    return {**base_context, **model_state}


def build_fit_context(
    base_context: BaseContext,
    model_state: ModelState,
    data_state: DataState,
) -> dict:
    return {**base_context, **model_state, **data_state}
