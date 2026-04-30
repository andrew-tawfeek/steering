from __future__ import annotations

from pathlib import Path
import re
import threading
import time

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, RichLog, Static

from .feature_cache import (
    CachedSource,
    FeatureCache,
    FeatureLabel,
    NeuronpediaDatasetClient,
    build_source_cache,
    default_feature_cache_path,
)
from .local_client import LocalServerClient, LocalServerError
from .neuronpedia_client import DEFAULT_STEERING_MODEL, NeuronpediaClient, summarize_feature
from .state import (
    SteerItem,
    SteeringState,
    SteeringError,
    clear_state,
    default_state_path,
    load_state,
    parse_layers,
    save_state,
    update_state,
)


CLEAR_CONFIRM_SECONDS = 4.0
STREAM_FLUSH_SECONDS = 0.08
STREAM_FLUSH_CHARS = 32
STREAM_PREVIEW_CHARS = 1800
DEFAULT_TEMPERATURE = 0.0
DEFAULT_UI_MAX_TOKENS = 32
COMPLETION_STOP_SEQUENCES = ("\r\n\r\n", "\n\n")
FORM_INPUT_IDS = frozenset({"feature-id", "strength", "layers", "sae-id", "label", "model-id"})
GENERATION_INPUT_IDS = frozenset({"max-tokens", "temperature"})
CACHE_INPUT_IDS = frozenset(
    {
        "cache-model-id",
        "cache-source",
        "cache-source-filter",
        "cache-query",
        "cache-feature-id",
    }
)
CACHE_SEARCH_LIMIT = 30


class SteeringTUI(App):
    TITLE = "Steering"
    SUB_TITLE = "Live feature steering"

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
        min-height: 0;
    }

    #chat-pane {
        width: 2fr;
        min-width: 56;
        padding: 1;
        border: solid $primary;
    }

    #steer-pane {
        width: 1fr;
        min-width: 54;
        padding: 1;
        border: solid $secondary;
    }

    #chat-log, #feature-log {
        height: 1fr;
        border: round $surface;
    }

    #stream-preview {
        display: none;
        max-height: 5;
        margin-bottom: 1;
        padding: 0 1;
        border: round $accent;
    }

    #steer-table {
        height: 10;
        margin-bottom: 1;
    }

    #source-table {
        height: 6;
        margin-bottom: 1;
    }

    #cache-results {
        height: 8;
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }

    .pane-title {
        text-style: bold;
        color: $accent;
    }

    .row {
        height: auto;
        margin-bottom: 1;
    }

    .inline-label, .field-label {
        width: auto;
        padding-right: 1;
        color: $text-muted;
    }

    .field-row {
        height: auto;
        margin-bottom: 1;
    }

    .field-label {
        width: 10;
        padding-top: 1;
    }

    .field-row Input {
        width: 1fr;
        margin-bottom: 0;
    }

    #generation-settings {
        height: auto;
        margin-bottom: 1;
    }

    #feature-cache-actions {
        height: auto;
        margin-bottom: 1;
    }

    #max-tokens {
        width: 9;
        margin-right: 1;
    }

    #temperature {
        width: 8;
    }

    Button {
        margin-right: 1;
    }

    #send {
        min-width: 9;
    }

    #status, #generation-status, #state-summary, #form-status, #cache-status {
        min-height: 1;
        color: $text-muted;
    }

    #form-status.error, #generation-status.error, #status.error, #cache-status.error {
        color: $error;
    }

    #form-status.success, #generation-status.success, #status.success, #cache-status.success {
        color: $success;
    }

    Input.invalid {
        border: heavy $error;
    }
    """

    BINDINGS = [
        ("f2", "focus_prompt", "Prompt"),
        ("f3", "focus_steers", "Steers"),
        ("f4", "focus_feature", "Feature"),
        ("f5", "refresh_state", "Refresh"),
        ("f6", "send_prompt", "Continue"),
        ("f7", "lookup_feature", "Lookup"),
        ("f8", "clear_steers", "Clear all"),
        ("f9", "check_backend", "Backend"),
        ("f10", "list_cache_sources", "Sources"),
        ("f11", "search_feature_cache", "Search"),
        ("f12", "apply_cached_feature", "Apply"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("ctrl+y", "list_cache_models", "Models"),
        ("ctrl+n", "new_steer", "New steer"),
        ("ctrl+o", "open_selected", "Edit row"),
        ("ctrl+e", "edit_selected", "Save edit"),
        ("delete", "remove_selected", "Remove"),
        ("escape", "clear_selection", "Deselect"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        server_url: str,
        max_tokens: int,
        temperature: float,
        state_path: Path | None,
        cache_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.client = LocalServerClient.from_env(server_url)
        self.neuronpedia = NeuronpediaClient.from_env()
        self.dataset_client = NeuronpediaDatasetClient()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.state_path = state_path
        self.cache_path = cache_path
        self._generating = False
        self._checking_backend = False
        self._lookup_running = False
        self._model_loading = False
        self._source_loading = False
        self._cache_downloading = False
        self._cache_searching = False
        self._cache_inspecting = False
        self._clear_confirm_pending = False
        self._backend_available: bool | None = None
        self._backend_model_name: str | None = None
        self._state_items: tuple[SteerItem, ...] = tuple()
        self._selected_index: int | None = None
        self._source_table_mode = "sources"
        self._cache_sources: tuple[str, ...] = tuple()
        self._cached_source_status: dict[tuple[str, str], CachedSource] = {}
        self._selected_cache_source: str | None = None
        self._cache_results: tuple[FeatureLabel, ...] = tuple()
        self._selected_cache_result_index: int | None = None
        self._stream_text = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="chat-pane"):
                yield Static("Backend: checking...", id="status")
                yield Label("Completion", classes="pane-title")
                with Horizontal(id="generation-settings"):
                    yield Label("Tokens", classes="inline-label")
                    yield Input(
                        str(self.max_tokens),
                        placeholder=str(DEFAULT_UI_MAX_TOKENS),
                        id="max-tokens",
                        name="Maximum new tokens",
                        tooltip="Maximum response tokens, 1 to 512.",
                        compact=True,
                    )
                    yield Label("Temp", classes="inline-label")
                    yield Input(
                        f"{self.temperature:g}",
                        placeholder=str(DEFAULT_TEMPERATURE),
                        id="temperature",
                        name="Temperature",
                        tooltip="Sampling temperature. Lower values make GPT-2 less random; use 0 for greedy output.",
                        compact=True,
                    )
                yield Static("Raw GPT-2 continuation mode.", id="generation-status")
                yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
                yield Static("", id="stream-preview")
                yield Input(
                    placeholder="Enter text for the model to continue...",
                    id="prompt",
                    name="Prompt",
                    tooltip="Press Enter or F6 to continue this text with the backend model.",
                )
                with Horizontal(classes="row"):
                    yield Button("Continue", id="send", variant="primary", tooltip="Continue the prompt with the backend model.")
                    yield Button("Health", id="check-health", tooltip="Refresh backend model status.")
                    yield Button("Clear Log", id="clear-chat", tooltip="Clear the generation log.")
            with VerticalScroll(id="steer-pane"):
                yield Label("Active Steers", classes="pane-title")
                yield DataTable(id="steer-table")
                yield Static("No active steers.", id="state-summary")
                with Horizontal(classes="field-row"):
                    yield Label("Feature", classes="field-label")
                    yield Input(
                        placeholder="required feature id",
                        id="feature-id",
                        name="Feature id",
                        tooltip="SAE feature index.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("Strength", classes="field-label")
                    yield Input(
                        placeholder="required strength",
                        id="strength",
                        name="Strength",
                        tooltip="Scalar multiplier for the SAE decoder vector.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("Layers", classes="field-label")
                    yield Input(
                        placeholder="required layer, e.g. 6",
                        id="layers",
                        name="Layers",
                        tooltip="Comma-separated GPT-2 layer shorthand.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("SAE Hook", classes="field-label")
                    yield Input(
                        placeholder="optional explicit SAE Lens hook",
                        id="sae-id",
                        name="SAE hook",
                        tooltip="Explicit SAE Lens hook id, used instead of layer shorthand.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("Model", classes="field-label")
                    yield Input(
                        placeholder=DEFAULT_STEERING_MODEL,
                        id="model-id",
                        name="Model id",
                        tooltip="Optional model id metadata. It must match the running backend model when set.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("Label", classes="field-label")
                    yield Input(
                        placeholder="optional note",
                        id="label",
                        name="Label",
                        tooltip="Optional local note stored with this steer.",
                    )
                yield Static("Ready.", id="form-status")
                with Horizontal(classes="row"):
                    yield Button("New", id="new-steer", tooltip="Clear the form and start a new steer.")
                    yield Button(
                        "Set Only",
                        id="replace",
                        variant="primary",
                        tooltip="Replace the active state with the form steer.",
                    )
                    yield Button("Add", id="append", tooltip="Append the form steer to the active state.")
                with Horizontal(classes="row"):
                    yield Button("Save Edit", id="edit-selected", tooltip="Save the form into the selected steer.")
                    yield Button(
                        "Remove",
                        id="remove-selected",
                        variant="error",
                        tooltip="Remove the selected steer.",
                    )
                    yield Button(
                        "Clear All",
                        id="clear-steers",
                        variant="error",
                        tooltip="Clear every active steer after confirmation.",
                    )
                with Horizontal(classes="row"):
                    yield Button("Lookup", id="lookup", tooltip="Load Neuronpedia metadata for this feature.")
                    yield Button("Refresh", id="refresh", tooltip="Reload active steers from disk.")
                yield Label("Feature Cache", classes="pane-title")
                with Horizontal(classes="field-row"):
                    yield Label("Model", classes="field-label")
                    yield Input(
                        DEFAULT_STEERING_MODEL,
                        placeholder=DEFAULT_STEERING_MODEL,
                        id="cache-model-id",
                        name="Cache model",
                        tooltip="Neuronpedia dataset model id.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("Filter", classes="field-label")
                    yield Input(
                        placeholder="optional source filter",
                        id="cache-source-filter",
                        name="Source filter",
                        tooltip="Optional text used when listing export sources.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("Source", classes="field-label")
                    yield Input(
                        placeholder="blank uses Layers, default 6-res-jb",
                        id="cache-source",
                        name="Cache source",
                        tooltip="Neuronpedia source id. Blank uses the steer form layers; with no layer, 6-res-jb.",
                    )
                with Horizontal(id="feature-cache-actions"):
                    yield Button("Models", id="list-models", tooltip="List available Neuronpedia export models.")
                    yield Button("Sources", id="list-sources", tooltip="List sources for the cache model.")
                    yield Button("Download", id="download-source", tooltip="Download labels for the selected source.")
                    yield Button("Cached", id="show-cached", tooltip="Show cached sources.")
                yield DataTable(id="source-table")
                with Horizontal(classes="field-row"):
                    yield Label("Search", classes="field-label")
                    yield Input(
                        placeholder="cached label search",
                        id="cache-query",
                        name="Feature search",
                        tooltip="Search cached labels. Press Enter or F11.",
                    )
                with Horizontal(classes="field-row"):
                    yield Label("Feature", classes="field-label")
                    yield Input(
                        placeholder="feature id",
                        id="cache-feature-id",
                        name="Cached feature id",
                        tooltip="Cached feature id to inspect.",
                    )
                with Horizontal(classes="row"):
                    yield Button("Search", id="search-cache", variant="primary", tooltip="Search cached labels.")
                    yield Button("Inspect", id="inspect-cache", tooltip="Show cached labels for one feature.")
                    yield Button("Apply", id="apply-cache", tooltip="Apply the selected cached feature to the steer form.")
                yield Static("Feature cache ready.", id="cache-status")
                yield DataTable(id="cache-results")
                yield RichLog(id="feature-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#steer-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.tooltip = "Use arrow keys to select a steer. Press Enter, Ctrl+E, Delete, or Esc for row actions."
        table.add_columns("#", "feature", "strength", "target", "label")
        source_table = self.query_one("#source-table", DataTable)
        source_table.cursor_type = "row"
        source_table.zebra_stripes = True
        source_table.tooltip = "Use arrow keys to choose a model or source. Press Enter to use it."
        source_table.add_columns("item", "cache")
        result_table = self.query_one("#cache-results", DataTable)
        result_table.cursor_type = "row"
        result_table.zebra_stripes = True
        result_table.tooltip = "Use arrow keys to choose a cached feature. Press Enter or F12 to apply it."
        result_table.add_columns("feature", "source", "label")
        self.query_one("#prompt", Input).focus()
        self.refresh_state()
        self.refresh_cache_status()
        self._check_health()
        self._sync_buttons()

    def action_refresh_state(self) -> None:
        self.refresh_state()

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", RichLog).clear()

    def action_send_prompt(self) -> None:
        self.send_prompt()

    def action_lookup_feature(self) -> None:
        self.lookup_feature()

    def action_clear_steers(self) -> None:
        self.clear_steers()

    def action_check_backend(self) -> None:
        self._check_health()

    def action_list_cache_sources(self) -> None:
        self.list_cache_sources()

    def action_list_cache_models(self) -> None:
        self.list_cache_models()

    def action_search_feature_cache(self) -> None:
        self.search_feature_cache()

    def action_apply_cached_feature(self) -> None:
        self.apply_cached_feature()

    def action_download_cache_source(self) -> None:
        self.download_cache_source()

    def action_inspect_cached_feature(self) -> None:
        self.inspect_cached_feature()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_focus_steers(self) -> None:
        table = self.query_one("#steer-table", DataTable)
        table.focus()
        if self._selected_index is None and self._state_items:
            self._select_steer(0)

    def action_focus_feature(self) -> None:
        self.query_one("#feature-id", Input).focus()

    def action_new_steer(self) -> None:
        self._cancel_clear_confirmation()
        self._selected_index = None
        self._clear_form()
        self._update_state_summary()
        self._set_form_status("New steer.", "success")
        self.query_one("#feature-id", Input).focus()
        self._sync_buttons()

    def action_open_selected(self) -> None:
        if self._selected_index is None:
            if not self._state_items:
                self._set_form_status("No active steer is available to edit.", "error")
                return
            self._select_steer(0)
        self.query_one("#feature-id", Input).focus()
        self._set_form_status(self._editing_text(), "success")

    def action_edit_selected(self) -> None:
        self.update_selected_steer()

    def action_remove_selected(self) -> None:
        self.remove_selected_steer()

    def action_clear_selection(self) -> None:
        self._cancel_clear_confirmation()
        self._selected_index = None
        self._clear_form()
        self._update_state_summary()
        self._set_form_status("Selection cleared.")
        self.query_one("#feature-id", Input).focus()
        self._sync_buttons()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter" and self.focused is self.query_one("#steer-table", DataTable):
            self.action_open_selected()
            event.stop()
        elif event.key == "enter" and self.focused is self.query_one("#source-table", DataTable):
            self.copy_selected_cache_source()
            event.stop()
        elif event.key == "enter" and self.focused is self.query_one("#cache-results", DataTable):
            self.apply_cached_feature()
            event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "prompt":
            self.send_prompt()
        elif event.input.id in FORM_INPUT_IDS:
            if self._selected_index is None:
                self.save_steer(append=False)
            else:
                self.update_selected_steer()
        elif event.input.id in GENERATION_INPUT_IDS:
            self.query_one("#prompt", Input).focus()
        elif event.input.id in {"cache-model-id", "cache-source-filter"}:
            self.list_cache_sources()
        elif event.input.id == "cache-source":
            self.download_cache_source()
        elif event.input.id == "cache-query":
            self.search_feature_cache()
        elif event.input.id == "cache-feature-id":
            self.inspect_cached_feature()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in FORM_INPUT_IDS:
            self._cancel_clear_confirmation()
            event.input.set_class(False, "invalid")
            self._set_form_status(self._editing_text())
        elif event.input.id == "max-tokens":
            if valid_token_count(event.input.value):
                event.input.set_class(False, "invalid")
                self._set_generation_ready()
        elif event.input.id == "temperature":
            if valid_temperature(event.input.value):
                event.input.set_class(False, "invalid")
                self._set_generation_ready()
        elif event.input.id in CACHE_INPUT_IDS:
            event.input.set_class(False, "invalid")
            self._set_cache_status("Feature cache ready.")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "steer-table":
            self._select_steer(event.cursor_row, table=event.data_table)
            self.action_open_selected()
        elif event.data_table.id == "source-table":
            self._select_cache_source(event.cursor_row, table=event.data_table)
            self.copy_selected_cache_source()
        elif event.data_table.id == "cache-results":
            self._select_cache_result(event.cursor_row, table=event.data_table)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "steer-table":
            self._select_steer(event.cursor_row, table=event.data_table)
        elif event.data_table.id == "source-table":
            self._select_cache_source(event.cursor_row, table=event.data_table)
        elif event.data_table.id == "cache-results":
            self._select_cache_result(event.cursor_row, table=event.data_table)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send":
            self.send_prompt()
        elif button_id == "clear-chat":
            self.action_clear_chat()
        elif button_id == "check-health":
            self._check_health()
        elif button_id == "new-steer":
            self.action_new_steer()
        elif button_id == "replace":
            self.save_steer(append=False)
        elif button_id == "append":
            self.save_steer(append=True)
        elif button_id == "edit-selected":
            self.update_selected_steer()
        elif button_id == "remove-selected":
            self.remove_selected_steer()
        elif button_id == "clear-steers":
            self.clear_steers()
        elif button_id == "lookup":
            self.lookup_feature()
        elif button_id == "refresh":
            self.refresh_state()
        elif button_id == "list-models":
            self.list_cache_models()
        elif button_id == "list-sources":
            self.list_cache_sources()
        elif button_id == "download-source":
            self.download_cache_source()
        elif button_id == "show-cached":
            self.refresh_cache_status(show_table=True)
        elif button_id == "search-cache":
            self.search_feature_cache()
        elif button_id == "inspect-cache":
            self.inspect_cached_feature()
        elif button_id == "apply-cache":
            self.apply_cached_feature()

    def refresh_state(self, *, select_index: int | None = None) -> None:
        self._cancel_clear_confirmation()
        table = self.query_one("#steer-table", DataTable)
        table.clear()
        try:
            state = load_state(self.state_path)
        except SteeringError as exc:
            self._state_items = tuple()
            self._selected_index = None
            self.query_one("#state-summary", Static).update("State file could not be read.")
            self._set_form_status(str(exc), "error")
            self._sync_buttons()
            return

        self._state_items = state.items
        for index, item in enumerate(self._state_items, start=1):
            table.add_row(
                str(index),
                str(item.feature_id),
                f"{item.strength:g}",
                format_target(item),
                item.label or "",
                key=f"steer-{index - 1}",
            )

        if self._state_items:
            wanted_index = self._selected_index if select_index is None else select_index
            if wanted_index is None:
                wanted_index = 0
            wanted_index = max(0, min(wanted_index, len(self._state_items) - 1))
            self._select_steer(wanted_index)
            self._update_state_summary()
        else:
            self._selected_index = None
            self.query_one("#state-summary", Static).update("No active steers.")
            self._set_form_status("Ready.")
            self._sync_buttons()

    def send_prompt(self) -> None:
        if self._generating:
            self._set_generation_status("Generation is already running.", "error")
            return
        if self._checking_backend:
            self._set_generation_status("Backend check is still running.", "error")
            self._sync_buttons()
            return
        if self._backend_available is False:
            self._set_generation_status("Backend unavailable. Check Health after starting the server.", "error")
            self._sync_buttons()
            return

        prompt_input = self.query_one("#prompt", Input)
        prompt = prompt_input.value.strip()
        if not prompt:
            self._set_generation_status("Enter text to continue.", "error")
            prompt_input.focus()
            return

        settings = self._read_generation_settings()
        if settings is None:
            return
        max_tokens, temperature = settings

        prompt_input.value = ""
        self.log_chat(f"[bold cyan]You:[/bold cyan] {prompt}")
        self._generating = True
        self.max_tokens = max_tokens
        self.temperature = temperature
        model = self._backend_model_name or "current model"
        self._start_stream_preview(model)
        self._set_generation_status(f"Generating with {model}...", "success")
        self._sync_buttons()
        threading.Thread(target=self._generate_thread, args=(prompt, max_tokens, temperature), daemon=True).start()

    def save_steer(self, *, append: bool) -> None:
        self._cancel_clear_confirmation()
        try:
            item = self._read_form_item()
            state = update_state(item, append=append, path=self.state_path)
        except SteeringError as exc:
            self._set_form_status(str(exc), "error")
            self.log_feature(f"[bold red]Invalid steer:[/bold red] {exc}")
            return

        select_index = len(state.items) - 1 if append else 0
        self.refresh_state(select_index=select_index)
        mode = "Appended" if append else "Replaced"
        self._set_form_status(f"{mode} steer {item.feature_id}.", "success")
        self.log_feature(f"[green]{mode} steer[/green]: feature {item.feature_id}, strength {item.strength:g}")

    def update_selected_steer(self) -> None:
        self._cancel_clear_confirmation()
        if self._selected_index is None:
            self._set_form_status("Select a steer before saving an edit.", "error")
            return

        try:
            item = self._read_form_item()
            state = load_state(self.state_path)
        except SteeringError as exc:
            self._set_form_status(str(exc), "error")
            self.log_feature(f"[bold red]Could not edit steer:[/bold red] {exc}")
            return

        if self._selected_index >= len(state.items):
            self._set_form_status("Selected steer no longer exists.", "error")
            self.refresh_state()
            return

        previous = state.items[self._selected_index]
        if previous.model_id and item.model_id is None:
            item = copy_item_with_model_id(item, previous.model_id)

        items = list(state.items)
        items[self._selected_index] = item
        save_state(SteeringState(items=tuple(items)), self.state_path)
        self.refresh_state(select_index=self._selected_index)
        self._set_form_status(f"Saved edit for steer {self._selected_index + 1}.", "success")
        self.log_feature(f"[green]Saved edit[/green]: steer {self._selected_index + 1}")

    def remove_selected_steer(self) -> None:
        self._cancel_clear_confirmation()
        if self._selected_index is None:
            self._set_form_status("Select a steer before removing it.", "error")
            return

        try:
            state = load_state(self.state_path)
        except SteeringError as exc:
            self._set_form_status(str(exc), "error")
            return

        if self._selected_index >= len(state.items):
            self._set_form_status("Selected steer no longer exists.", "error")
            self.refresh_state()
            return

        removed = state.items[self._selected_index]
        items = list(state.items)
        del items[self._selected_index]
        save_state(SteeringState(items=tuple(items)), self.state_path)
        next_index = None if not items else min(self._selected_index, len(items) - 1)
        self.refresh_state(select_index=next_index)
        if not items:
            self._clear_form()
        self._set_form_status(f"Removed feature {removed.feature_id}.", "success")
        self.log_feature(f"[bold red]Removed steer[/bold red]: feature {removed.feature_id}")

    def clear_steers(self) -> None:
        if not self._state_items:
            self._cancel_clear_confirmation()
            self._set_form_status("No active steers to clear.")
            return
        if not self._clear_confirm_pending:
            count = len(self._state_items)
            noun = "steer" if count == 1 else "steers"
            self._clear_confirm_pending = True
            self._set_clear_button_label(confirming=True)
            self._set_form_status(f"Press Clear All again to clear {count} active {noun}.", "error")
            self.set_timer(CLEAR_CONFIRM_SECONDS, self._expire_clear_confirmation)
            return

        self._cancel_clear_confirmation()
        clear_state(self.state_path)
        self.refresh_state()
        self._clear_form()
        self._set_form_status("Cleared all active steers.", "success")
        self.log_feature("[bold red]Cleared all active steers.[/bold red]")

    def lookup_feature(self) -> None:
        if self._lookup_running:
            self._set_form_status("Feature lookup is already running.", "error")
            return

        try:
            feature_id, sae_id = self._read_lookup_target()
        except SteeringError as exc:
            self._set_form_status("Lookup needs a feature id and layer.", "error")
            self.log_feature(f"[bold red]Lookup needs a feature id and layer:[/bold red] {exc}")
            return

        self._lookup_running = True
        self._set_form_status(f"Looking up feature {feature_id}.", "success")
        self._sync_buttons()
        self.log_feature(f"[dim]Looking up {DEFAULT_STEERING_MODEL}/{sae_id}/{feature_id}...[/dim]")
        threading.Thread(target=self._lookup_thread, args=(sae_id, feature_id), daemon=True).start()

    def refresh_cache_status(self, *, show_table: bool = False) -> None:
        try:
            rows = self._feature_cache().status()
        except Exception as exc:
            self._set_cache_status(f"Feature cache unavailable: {exc}", "error")
            return

        self._cached_source_status = {(row.model_id, row.source_id): row for row in rows}
        cache_path = self.cache_path or default_feature_cache_path()
        if rows:
            self._set_cache_status(f"{len(rows)} cached source(s) at {cache_path}.", "success")
        else:
            self._set_cache_status(f"No cached sources at {cache_path}.")

        if show_table:
            model_id = self.query_one("#cache-model-id", Input).value.strip()
            if model_id:
                sources = [row.source_id for row in rows if row.model_id == model_id]
                self._populate_source_table(model_id, sources, mode="sources")
            else:
                self._populate_source_table(
                    None,
                    [f"{row.model_id}/{row.source_id}" for row in rows],
                    mode="cached",
                )

    def list_cache_models(self) -> None:
        if self._model_loading:
            self._set_cache_status("Model listing is already running.", "error")
            return

        self._model_loading = True
        self._set_cache_status("Listing Neuronpedia export models...", "success")
        self._sync_buttons()
        threading.Thread(target=self._list_models_thread, daemon=True).start()

    def list_cache_sources(self) -> None:
        if self._source_loading:
            self._set_cache_status("Source listing is already running.", "error")
            return

        try:
            model_id = self._read_cache_model()
        except SteeringError as exc:
            self._set_cache_status(str(exc), "error")
            return

        contains = self.query_one("#cache-source-filter", Input).value.strip()
        self._source_loading = True
        self._set_cache_status(f"Listing sources for {model_id}...", "success")
        self._sync_buttons()
        threading.Thread(target=self._list_sources_thread, args=(model_id, contains), daemon=True).start()

    def download_cache_source(self) -> None:
        if self._cache_downloading:
            self._set_cache_status("Feature cache download is already running.", "error")
            return

        try:
            model_id = self._read_cache_model()
            source_id = self._read_cache_source(required=True)
        except SteeringError as exc:
            self._set_cache_status(str(exc), "error")
            return

        self._cache_downloading = True
        self._set_cache_status(f"Downloading labels for {model_id}/{source_id}...", "success")
        self._sync_buttons()
        self.log_feature(f"[dim]Caching labels for {model_id}/{source_id}...[/dim]")
        threading.Thread(target=self._download_source_thread, args=(model_id, source_id), daemon=True).start()

    def search_feature_cache(self) -> None:
        if self._cache_searching:
            self._set_cache_status("Feature cache search is already running.", "error")
            return

        query_input = self.query_one("#cache-query", Input)
        query = query_input.value.strip()
        if not query:
            query_input.set_class(True, "invalid")
            query_input.focus()
            self._set_cache_status("Search query is required.", "error")
            return

        try:
            model_id = self._read_cache_model(allow_blank=True)
            source_id = self._read_cache_source(required=False)
        except SteeringError as exc:
            self._set_cache_status(str(exc), "error")
            return

        self._cache_searching = True
        self._set_cache_status(f"Searching cached labels for {query!r}...", "success")
        self._sync_buttons()
        threading.Thread(target=self._search_cache_thread, args=(query, model_id, source_id), daemon=True).start()

    def inspect_cached_feature(self) -> None:
        if self._cache_inspecting:
            self._set_cache_status("Feature inspection is already running.", "error")
            return

        try:
            model_id, source_id, feature_id = self._read_cached_feature_target()
        except SteeringError as exc:
            self._set_cache_status(str(exc), "error")
            return

        self._cache_inspecting = True
        self._set_cache_status(f"Inspecting cached feature {feature_id}...", "success")
        self._sync_buttons()
        threading.Thread(
            target=self._inspect_cached_feature_thread,
            args=(model_id, source_id, feature_id),
            daemon=True,
        ).start()

    def apply_cached_feature(self) -> None:
        label = self._selected_cache_label()
        if label is None:
            self._set_cache_status("Select a cached feature before applying it.", "error")
            return

        layers, sae_id = steer_target_from_neuronpedia_source(label.source_id)
        self.query_one("#feature-id", Input).value = str(label.feature_id)
        self.query_one("#layers", Input).value = layers
        self.query_one("#sae-id", Input).value = sae_id
        self.query_one("#model-id", Input).value = label.model_id
        self.query_one("#label", Input).value = compact_whitespace(label.description)

        strength_input = self.query_one("#strength", Input)
        if not strength_input.value.strip():
            strength_input.value = "10"

        self._clear_validation()
        self._set_form_status(f"Applied cached feature {label.feature_id} to the steer form.", "success")
        self._set_cache_status(f"Applied {label.model_id}/{label.source_id}/{label.feature_id}.", "success")
        self.log_feature(
            f"[green]Applied cached feature[/green]: "
            f"{label.model_id}/{label.source_id}/{label.feature_id}"
        )
        self.query_one("#feature-id", Input).focus()

    def copy_selected_cache_source(self) -> None:
        if self._selected_cache_source is None:
            self._set_cache_status("No model or source is selected.", "error")
            return

        if self._source_table_mode == "models":
            self.query_one("#cache-model-id", Input).value = self._selected_cache_source
            self.query_one("#cache-source", Input).value = ""
            self._set_cache_status(f"Selected model {self._selected_cache_source}. Listing sources...", "success")
            self.list_cache_sources()
            return

        if self._source_table_mode == "cached" and "/" in self._selected_cache_source:
            model_id, source_id = self._selected_cache_source.split("/", 1)
            self.query_one("#cache-model-id", Input).value = model_id
            self.query_one("#cache-source", Input).value = source_id
            self._set_cache_status(f"Selected cached source {model_id}/{source_id}.", "success")
            return

        self.query_one("#cache-source", Input).value = self._selected_cache_source
        self._set_cache_status(f"Selected source {self._selected_cache_source}.", "success")

    def _check_health(self) -> None:
        if self._checking_backend:
            return
        self._checking_backend = True
        self._set_backend_status("Backend: checking...")
        self._sync_buttons()
        threading.Thread(target=self._health_thread, daemon=True).start()

    def _health_thread(self) -> None:
        try:
            health = self.client.health()
            model_name = str(health.get("model_name", "unknown"))
            text = (
                f"Backend: {model_name} raw completion "
                f"on {health.get('device', 'unknown')} | {self.client.base_url}"
            )
        except Exception as exc:
            text = f"Backend unavailable: {exc}"
            self._call_from_worker(self._apply_health_result, text, "error", None)
            return
        self._call_from_worker(self._apply_health_result, text, "success", model_name)

    def _generate_thread(self, prompt: str, max_tokens: int, temperature: float) -> None:
        chunks = None
        try:
            chunks = self.client.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                seed=None,
                stream=True,
            )
            parts: list[str] = []
            last_preview_text = ""
            last_flush = time.monotonic()
            stopped_at_paragraph = False
            for chunk in chunks:
                parts.append(chunk)
                raw_text = "".join(parts)
                preview_text, stopped = completion_text_for_ui(raw_text)
                now = time.monotonic()
                if (
                    preview_text != last_preview_text
                    and (
                        not last_preview_text
                        or len(preview_text) - len(last_preview_text) >= STREAM_FLUSH_CHARS
                        or stopped
                    )
                    or now - last_flush >= STREAM_FLUSH_SECONDS
                ):
                    self._call_from_worker(self._set_stream_text, preview_text)
                    last_preview_text = preview_text
                    last_flush = now
                if stopped:
                    stopped_at_paragraph = True
                    if preview_text != last_preview_text:
                        self._call_from_worker(self._set_stream_text, preview_text)
                        last_preview_text = preview_text
                    break

            text, stopped = completion_text_for_ui("".join(parts))
            stopped_at_paragraph = stopped_at_paragraph or stopped
            if text != last_preview_text:
                self._call_from_worker(self._set_stream_text, text)
            text = text.strip()
            if not text:
                text = "[dim](empty response)[/dim]"
            self._call_from_worker(self.log_chat, f"[bold green]Completion:[/bold green] {text}")
            status = "Stopped at first paragraph." if stopped_at_paragraph else "Raw completion ready."
            self._call_from_worker(self._set_generation_status, status, "success")
        except LocalServerError as exc:
            self._call_from_worker(self.log_chat, f"[bold red]Generation failed:[/bold red] {exc}")
            self._call_from_worker(self._set_generation_status, str(exc), "error")
            self._call_from_worker(self._apply_health_result, f"Backend unavailable: {exc}", "error", None)
        except Exception as exc:
            self._call_from_worker(self.log_chat, f"[bold red]Generation failed:[/bold red] {exc}")
            self._call_from_worker(self._set_generation_status, str(exc), "error")
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
            self._call_from_worker(self._finish_generation)

    def _lookup_thread(self, sae_id: str, feature_id: int) -> None:
        try:
            data = self.neuronpedia.feature(DEFAULT_STEERING_MODEL, sae_id, feature_id)
            text = summarize_feature(data)
            self._call_from_worker(self._apply_lookup_result, text, data, feature_id)
        except Exception as exc:
            self._call_from_worker(self.log_feature, f"[bold red]Lookup failed:[/bold red] {exc}")
            self._call_from_worker(self._set_form_status, f"Lookup failed: {exc}", "error")
        finally:
            self._call_from_worker(self._finish_lookup)

    def _list_models_thread(self) -> None:
        try:
            models = self.dataset_client.list_models()
            self._call_from_worker(self._apply_model_list, models)
        except Exception as exc:
            self._call_from_worker(self.log_feature, f"[bold red]Model listing failed:[/bold red] {exc}")
            self._call_from_worker(self._set_cache_status, f"Model listing failed: {exc}", "error")
        finally:
            self._call_from_worker(self._finish_model_listing)

    def _list_sources_thread(self, model_id: str, contains: str) -> None:
        try:
            sources = self.dataset_client.list_sources(model_id)
            if contains:
                needle = contains.casefold()
                sources = [source for source in sources if needle in source.casefold()]
            rows = self._feature_cache().status()
            self._call_from_worker(self._apply_source_list, model_id, sources, rows)
        except Exception as exc:
            self._call_from_worker(self.log_feature, f"[bold red]Source listing failed:[/bold red] {exc}")
            self._call_from_worker(self._set_cache_status, f"Source listing failed: {exc}", "error")
        finally:
            self._call_from_worker(self._finish_source_listing)

    def _download_source_thread(self, model_id: str, source_id: str) -> None:
        try:
            cached = build_source_cache(
                model_id=model_id,
                source_id=source_id,
                cache_path=self.cache_path,
                dataset_client=self.dataset_client,
            )
            self._call_from_worker(self._apply_download_result, cached)
        except Exception as exc:
            self._call_from_worker(self.log_feature, f"[bold red]Cache download failed:[/bold red] {exc}")
            self._call_from_worker(self._set_cache_status, f"Cache download failed: {exc}", "error")
        finally:
            self._call_from_worker(self._finish_cache_download)

    def _search_cache_thread(self, query: str, model_id: str | None, source_id: str | None) -> None:
        try:
            labels = self._feature_cache().search(
                query,
                model_id=model_id or None,
                source_id=source_id or None,
                limit=CACHE_SEARCH_LIMIT,
            )
            self._call_from_worker(self._apply_cache_results, labels, f"Search matched {len(labels)} feature label(s).")
        except Exception as exc:
            self._call_from_worker(self.log_feature, f"[bold red]Cache search failed:[/bold red] {exc}")
            self._call_from_worker(self._set_cache_status, f"Cache search failed: {exc}", "error")
        finally:
            self._call_from_worker(self._finish_cache_search)

    def _inspect_cached_feature_thread(self, model_id: str, source_id: str, feature_id: int) -> None:
        try:
            labels = self._feature_cache().get(model_id=model_id, source_id=source_id, feature_id=feature_id)
            self._call_from_worker(
                self._apply_cache_results,
                labels,
                f"Feature {feature_id} has {len(labels)} cached label(s).",
            )
            if labels:
                self._call_from_worker(self._log_cached_feature_labels, labels)
        except Exception as exc:
            self._call_from_worker(self.log_feature, f"[bold red]Feature inspection failed:[/bold red] {exc}")
            self._call_from_worker(self._set_cache_status, f"Feature inspection failed: {exc}", "error")
        finally:
            self._call_from_worker(self._finish_cache_inspect)

    def _call_from_worker(self, callback, *args) -> None:
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                return
            raise

    def log_chat(self, message: str) -> None:
        try:
            self.query_one("#chat-log", RichLog).write(message)
        except NoMatches:
            return

    def log_feature(self, message: str) -> None:
        try:
            self.query_one("#feature-log", RichLog).write(message)
        except NoMatches:
            return

    def _feature_cache(self) -> FeatureCache:
        return FeatureCache(self.cache_path)

    def _read_cache_model(self, *, allow_blank: bool = False) -> str:
        model_input = self.query_one("#cache-model-id", Input)
        model_input.set_class(False, "invalid")
        model_id = model_input.value.strip()
        if not model_id and not allow_blank:
            model_input.set_class(True, "invalid")
            model_input.focus()
            raise SteeringError("Cache model is required.")
        return model_id

    def _read_cache_source(self, *, required: bool) -> str | None:
        source_input = self.query_one("#cache-source", Input)
        source_input.set_class(False, "invalid")
        source_id = source_input.value.strip()
        if source_id:
            return source_id
        if required:
            try:
                source_id = self._infer_cache_source()
            except SteeringError:
                source_input.set_class(True, "invalid")
                source_input.focus()
                raise
            source_input.value = source_id
            self._set_cache_status(f"Using inferred source {source_id}.", "success")
            return source_id
        return None

    def _infer_cache_source(self) -> str:
        layers_input = self.query_one("#layers", Input)
        sae_input = self.query_one("#sae-id", Input)
        try:
            return neuronpedia_sae_id_from_form(layers_input.value, sae_input.value)
        except SteeringError as exc:
            raise SteeringError("Cache source is required, or provide a valid layer/SAE hook to infer it.") from exc

    def _read_cached_feature_target(self) -> tuple[str, str, int]:
        selected = self._selected_cache_label()
        try:
            model_id = self._read_cache_model(allow_blank=selected is not None)
            source_id = self._read_cache_source(required=selected is None)
        except SteeringError:
            raise

        feature_input = self.query_one("#cache-feature-id", Input)
        feature_input.set_class(False, "invalid")
        raw_feature_id = feature_input.value.strip()
        if not raw_feature_id and selected is not None:
            raw_feature_id = str(selected.feature_id)
        if not raw_feature_id:
            feature_input.set_class(True, "invalid")
            feature_input.focus()
            raise SteeringError("Cached feature id is required.")

        try:
            feature_id = int(raw_feature_id)
        except ValueError as exc:
            feature_input.set_class(True, "invalid")
            feature_input.focus()
            raise SteeringError("Cached feature id must be a whole number.") from exc
        if feature_id < 0:
            feature_input.set_class(True, "invalid")
            feature_input.focus()
            raise SteeringError("Cached feature id must be >= 0.")

        if selected is not None:
            model_id = model_id or selected.model_id
            source_id = source_id or selected.source_id
        if not model_id or not source_id:
            raise SteeringError("Cache model and source are required.")
        return model_id, source_id, feature_id

    def _apply_model_list(self, models: list[str]) -> None:
        self._populate_source_table(None, models, mode="models")
        count = len(models)
        noun = "model" if count == 1 else "models"
        self._set_cache_status(f"Found {count} export {noun}.", "success" if models else None)
        if models:
            self._select_cache_source(0)

    def _apply_source_list(self, model_id: str, sources: list[str], rows: list[CachedSource]) -> None:
        self._cached_source_status = {(row.model_id, row.source_id): row for row in rows}
        self._populate_source_table(model_id, sources, mode="sources")
        count = len(sources)
        noun = "source" if count == 1 else "sources"
        self._set_cache_status(f"Found {count} {noun} for {model_id}.", "success" if sources else None)
        if sources:
            self._select_cache_source(0)

    def _apply_download_result(self, cached: CachedSource) -> None:
        self._cached_source_status[(cached.model_id, cached.source_id)] = cached
        self._set_cache_status(
            f"Cached {cached.label_count} label(s) for {cached.model_id}/{cached.source_id}.",
            "success",
        )
        self.log_feature(
            f"[green]Cached labels[/green]: {cached.label_count} label(s), "
            f"{cached.feature_count} feature(s) for {cached.model_id}/{cached.source_id}"
        )
        self._refresh_visible_source_cache_markers(cached.model_id)

    def _apply_cache_results(self, labels: list[FeatureLabel], message: str) -> None:
        self._cache_results = tuple(labels)
        self._selected_cache_result_index = None
        table = self.query_one("#cache-results", DataTable)
        table.clear()
        for index, label in enumerate(labels):
            table.add_row(
                str(label.feature_id),
                label.source_id,
                compact_label(label.description),
                key=f"cache-result-{index}",
            )

        if labels:
            self._select_cache_result(0)
            state = "success"
        else:
            state = None
        self._set_cache_status(message, state)

    def _log_cached_feature_labels(self, labels: list[FeatureLabel]) -> None:
        for label in labels:
            self.log_feature(format_cached_label(label))

    def _populate_source_table(self, model_id: str | None, sources: list[str], *, mode: str) -> None:
        self._source_table_mode = mode
        self._cache_sources = tuple(sources)
        self._selected_cache_source = None
        table = self.query_one("#source-table", DataTable)
        table.clear()
        for index, source in enumerate(sources):
            marker = "-"
            source_model = model_id
            source_id = source
            if model_id is None and "/" in source:
                source_model, source_id = source.split("/", 1)
            if source_model is not None:
                cached = self._cached_source_status.get((source_model, source_id))
                if cached is not None:
                    marker = f"{cached.label_count} labels"
            table.add_row(source, marker, key=f"cache-source-{index}")

    def _refresh_visible_source_cache_markers(self, model_id: str) -> None:
        if not self._cache_sources:
            return
        selected = self._selected_cache_source
        self._populate_source_table(model_id, list(self._cache_sources), mode=self._source_table_mode)
        if selected in self._cache_sources:
            self._select_cache_source(self._cache_sources.index(selected))

    def _select_cache_source(self, row_index: int, *, table: DataTable | None = None) -> None:
        if row_index < 0 or row_index >= len(self._cache_sources):
            return
        self._selected_cache_source = self._cache_sources[row_index]
        try:
            table = table or self.query_one("#source-table", DataTable)
            if table.row_count and table.cursor_row != row_index:
                table.move_cursor(row=row_index, column=0, animate=False)
            if self._source_table_mode == "models":
                self.query_one("#cache-model-id", Input).value = self._selected_cache_source
                self.query_one("#cache-source", Input).value = ""
            elif self._source_table_mode == "cached" and "/" in self._selected_cache_source:
                model_id, source_id = self._selected_cache_source.split("/", 1)
                self.query_one("#cache-model-id", Input).value = model_id
                self.query_one("#cache-source", Input).value = source_id
            else:
                self.query_one("#cache-source", Input).value = self._selected_cache_source
        except NoMatches:
            return

    def _select_cache_result(self, row_index: int, *, table: DataTable | None = None) -> None:
        if row_index < 0 or row_index >= len(self._cache_results):
            return
        self._selected_cache_result_index = row_index
        label = self._cache_results[row_index]
        try:
            table = table or self.query_one("#cache-results", DataTable)
            if table.row_count and table.cursor_row != row_index:
                table.move_cursor(row=row_index, column=0, animate=False)
            self.query_one("#cache-model-id", Input).value = label.model_id
            self.query_one("#cache-source", Input).value = label.source_id
            self.query_one("#cache-feature-id", Input).value = str(label.feature_id)
        except NoMatches:
            return
        self._sync_buttons()

    def _selected_cache_label(self) -> FeatureLabel | None:
        if self._selected_cache_result_index is None:
            return None
        if self._selected_cache_result_index >= len(self._cache_results):
            return None
        return self._cache_results[self._selected_cache_result_index]

    def _read_generation_settings(self) -> tuple[int, float] | None:
        tokens_input = self.query_one("#max-tokens", Input)
        temperature_input = self.query_one("#temperature", Input)
        tokens_input.set_class(False, "invalid")
        temperature_input.set_class(False, "invalid")

        try:
            max_tokens = int(tokens_input.value.strip())
            if not 1 <= max_tokens <= 512:
                raise SteeringError("tokens must be between 1 and 512")
        except SteeringError as exc:
            tokens_input.set_class(True, "invalid")
            self._set_generation_status(str(exc), "error")
            tokens_input.focus()
            return None
        except ValueError:
            tokens_input.set_class(True, "invalid")
            self._set_generation_status("Tokens must be a whole number.", "error")
            tokens_input.focus()
            return None

        try:
            temperature = float(temperature_input.value.strip())
            if temperature < 0:
                raise SteeringError("temperature must be >= 0")
        except SteeringError as exc:
            temperature_input.set_class(True, "invalid")
            self._set_generation_status(str(exc), "error")
            temperature_input.focus()
            return None
        except ValueError:
            temperature_input.set_class(True, "invalid")
            self._set_generation_status("Temperature must be a number.", "error")
            temperature_input.focus()
            return None

        return max_tokens, temperature

    def _read_lookup_target(self) -> tuple[int, str]:
        feature_input = self.query_one("#feature-id", Input)
        layers_input = self.query_one("#layers", Input)
        sae_input = self.query_one("#sae-id", Input)

        for input_widget in (feature_input, layers_input, sae_input):
            input_widget.set_class(False, "invalid")

        try:
            feature_id = int(feature_input.value.strip())
        except ValueError as exc:
            feature_input.set_class(True, "invalid")
            feature_input.focus()
            raise SteeringError("feature id must be a whole number") from exc
        if feature_id < 0:
            feature_input.set_class(True, "invalid")
            feature_input.focus()
            raise SteeringError("feature id must be >= 0")

        try:
            sae_id = neuronpedia_sae_id_from_form(layers_input.value, sae_input.value)
        except SteeringError:
            layers_input.set_class(True, "invalid")
            sae_input.set_class(True, "invalid")
            layers_input.focus()
            raise

        return feature_id, sae_id

    def _read_form_item(self) -> SteerItem:
        feature_input = self.query_one("#feature-id", Input)
        strength_input = self.query_one("#strength", Input)
        layers_input = self.query_one("#layers", Input)
        sae_input = self.query_one("#sae-id", Input)
        model_input = self.query_one("#model-id", Input)
        label_input = self.query_one("#label", Input)

        for input_widget in (feature_input, strength_input, layers_input, sae_input, model_input):
            input_widget.set_class(False, "invalid")

        try:
            feature_raw = feature_input.value.strip()
            if not feature_raw:
                raise SteeringError("feature id is required")
            feature_id = int(feature_raw)
            if feature_id < 0:
                raise SteeringError("feature id must be >= 0")
        except SteeringError:
            feature_input.set_class(True, "invalid")
            feature_input.focus()
            raise
        except ValueError as exc:
            feature_input.set_class(True, "invalid")
            feature_input.focus()
            raise SteeringError("feature id must be a whole number") from exc

        try:
            strength_raw = strength_input.value.strip()
            if not strength_raw:
                raise SteeringError("strength is required")
            strength = float(strength_raw)
        except SteeringError:
            strength_input.set_class(True, "invalid")
            strength_input.focus()
            raise
        except ValueError as exc:
            strength_input.set_class(True, "invalid")
            strength_input.focus()
            raise SteeringError("strength must be a number") from exc

        layers_raw = layers_input.value.strip()
        sae_id = sae_input.value.strip() or None
        try:
            layers = parse_layers(layers_raw) if layers_raw else tuple()
            item = SteerItem(
                feature_id=feature_id,
                strength=strength,
                layers=layers,
                sae_id=sae_id,
                model_id=model_input.value.strip() or None,
                label=label_input.value.strip() or None,
            )
        except SteeringError:
            layers_input.set_class(True, "invalid")
            if not layers_raw:
                sae_input.set_class(True, "invalid")
            if model_input.value.strip() == "":
                model_input.set_class(False, "invalid")
            layers_input.focus()
            raise

        return item

    def _select_steer(self, row_index: int, *, table: DataTable | None = None) -> None:
        if row_index < 0 or row_index >= len(self._state_items):
            return

        self._cancel_clear_confirmation()
        self._selected_index = row_index
        try:
            table = table or self.query_one("#steer-table", DataTable)
        except NoMatches:
            return
        if table.row_count and table.cursor_row != row_index:
            table.move_cursor(row=row_index, column=0, animate=False)
        try:
            self._populate_form(self._state_items[row_index])
            self._set_form_status(self._selection_text())
            self._update_state_summary()
            self._sync_buttons()
        except NoMatches:
            return

    def _populate_form(self, item: SteerItem) -> None:
        self.query_one("#feature-id", Input).value = str(item.feature_id)
        self.query_one("#strength", Input).value = f"{item.strength:g}"
        self.query_one("#layers", Input).value = ",".join(str(layer) for layer in item.layers)
        self.query_one("#sae-id", Input).value = item.sae_id or ""
        self.query_one("#model-id", Input).value = item.model_id or ""
        self.query_one("#label", Input).value = item.label or ""
        self._clear_validation()

    def _clear_form(self) -> None:
        for selector in ("#feature-id", "#strength", "#layers", "#sae-id", "#model-id", "#label"):
            self.query_one(selector, Input).value = ""
        self._clear_validation()

    def _clear_validation(self) -> None:
        for selector in (
            "#feature-id",
            "#strength",
            "#layers",
            "#sae-id",
            "#model-id",
            "#label",
            "#max-tokens",
            "#temperature",
            "#cache-model-id",
            "#cache-source",
            "#cache-source-filter",
            "#cache-query",
            "#cache-feature-id",
        ):
            self.query_one(selector, Input).set_class(False, "invalid")

    def _selection_text(self) -> str:
        if self._selected_index is None:
            return "Ready."
        return f"Selected steer {self._selected_index + 1} of {len(self._state_items)}."

    def _editing_text(self) -> str:
        if self._selected_index is None:
            return "Ready."
        return f"Editing steer {self._selected_index + 1}. Save Edit keeps these changes."

    def _apply_health_result(self, message: str, state: str, model_name: str | None) -> None:
        self._checking_backend = False
        self._backend_available = state == "success"
        self._backend_model_name = model_name
        self._set_backend_status(message, state)
        if not self._generating:
            self._set_generation_ready()
        self._sync_buttons()

    def _apply_lookup_result(self, message: str, data: dict, feature_id: int) -> None:
        self.log_feature(message)
        self._prefill_lookup_metadata(data)
        self._set_form_status(f"Lookup loaded feature {feature_id}.", "success")

    def _set_backend_status(self, message: str, state: str | None = None) -> None:
        try:
            status = self.query_one("#status", Static)
        except NoMatches:
            return
        status.update(message)
        status.set_class(state == "error", "error")
        status.set_class(state == "success", "success")

    def _set_generation_status(self, message: str, state: str | None = None) -> None:
        try:
            status = self.query_one("#generation-status", Static)
        except NoMatches:
            return
        status.update(message)
        status.set_class(state == "error", "error")
        status.set_class(state == "success", "success")

    def _set_generation_ready(self) -> None:
        if self._backend_available is False:
            self._set_generation_status("Backend unavailable.", "error")
        else:
            self._set_generation_status(
                "Raw completion ready." if self._backend_available else "Raw GPT-2 continuation mode.",
                "success" if self._backend_available else None,
            )

    def _set_form_status(self, message: str, state: str | None = None) -> None:
        try:
            status = self.query_one("#form-status", Static)
        except NoMatches:
            return
        status.update(message)
        status.set_class(state == "error", "error")
        status.set_class(state == "success", "success")

    def _set_cache_status(self, message: str, state: str | None = None) -> None:
        try:
            status = self.query_one("#cache-status", Static)
        except NoMatches:
            return
        status.update(message)
        status.set_class(state == "error", "error")
        status.set_class(state == "success", "success")

    def _start_stream_preview(self, model_name: str) -> None:
        self._stream_text = ""
        try:
            preview = self.query_one("#stream-preview", Static)
        except NoMatches:
            return
        preview.display = True
        preview.update(Text(f"{model_name} is preparing the next token..."))

    def _append_stream_chunk(self, chunk: str) -> None:
        self._set_stream_text(self._stream_text + chunk)

    def _set_stream_text(self, text: str) -> None:
        self._stream_text = text
        visible_text = self._stream_text.strip() or self._stream_text
        if len(visible_text) > STREAM_PREVIEW_CHARS:
            visible_text = "..." + visible_text[-STREAM_PREVIEW_CHARS:]
        try:
            self.query_one("#stream-preview", Static).update(Text(visible_text or "Waiting for first token..."))
        except NoMatches:
            return

    def _hide_stream_preview(self) -> None:
        try:
            preview = self.query_one("#stream-preview", Static)
        except NoMatches:
            return
        preview.display = False
        preview.update("")

    def _finish_generation(self) -> None:
        self._generating = False
        self._hide_stream_preview()
        self._sync_buttons()

    def _finish_lookup(self) -> None:
        self._lookup_running = False
        self._sync_buttons()

    def _finish_model_listing(self) -> None:
        self._model_loading = False
        self._sync_buttons()

    def _finish_source_listing(self) -> None:
        self._source_loading = False
        self._sync_buttons()

    def _finish_cache_download(self) -> None:
        self._cache_downloading = False
        self._sync_buttons()

    def _finish_cache_search(self) -> None:
        self._cache_searching = False
        self._sync_buttons()

    def _finish_cache_inspect(self) -> None:
        self._cache_inspecting = False
        self._sync_buttons()

    def _cancel_clear_confirmation(self) -> None:
        if not self._clear_confirm_pending:
            return
        self._clear_confirm_pending = False
        self._set_clear_button_label(confirming=False)

    def _expire_clear_confirmation(self) -> None:
        if not self._clear_confirm_pending:
            return
        self._clear_confirm_pending = False
        self._set_clear_button_label(confirming=False)
        self._set_form_status(self._selection_text())

    def _set_clear_button_label(self, *, confirming: bool) -> None:
        try:
            button = self.query_one("#clear-steers", Button)
        except NoMatches:
            return
        button.label = "Confirm Clear" if confirming else "Clear All"

    def _prefill_lookup_metadata(self, data: dict) -> None:
        try:
            label_input = self.query_one("#label", Input)
            strength_input = self.query_one("#strength", Input)
        except NoMatches:
            return

        if not label_input.value.strip():
            label = feature_label_from_data(data)
            if label:
                label_input.value = label

        if not strength_input.value.strip():
            strength = data.get("vectorDefaultSteerStrength")
            if strength is not None:
                strength_input.value = format_lookup_strength(strength)

    def _sync_buttons(self) -> None:
        has_selection = self._selected_index is not None and self._selected_index < len(self._state_items)
        has_steers = bool(self._state_items)
        backend_unavailable = self._backend_available is False
        try:
            self.query_one("#send", Button).disabled = self._generating or self._checking_backend or backend_unavailable
            self.query_one("#check-health", Button).disabled = self._checking_backend
            self.query_one("#edit-selected", Button).disabled = not has_selection
            self.query_one("#remove-selected", Button).disabled = not has_selection
            self.query_one("#clear-steers", Button).disabled = not has_steers
            self.query_one("#lookup", Button).disabled = self._lookup_running
            self.query_one("#list-models", Button).disabled = self._model_loading
            self.query_one("#list-sources", Button).disabled = self._source_loading
            self.query_one("#download-source", Button).disabled = self._cache_downloading
            self.query_one("#show-cached", Button).disabled = (
                self._source_loading or self._cache_downloading or self._cache_searching
            )
            self.query_one("#search-cache", Button).disabled = self._cache_searching
            self.query_one("#inspect-cache", Button).disabled = self._cache_inspecting
            self.query_one("#apply-cache", Button).disabled = self._selected_cache_label() is None
        except NoMatches:
            return

    def _update_state_summary(self) -> None:
        try:
            summary = self.query_one("#state-summary", Static)
        except NoMatches:
            return
        if not self._state_items:
            summary.update("No active steers.")
            return

        count = len(self._state_items)
        noun = "steer" if count == 1 else "steers"
        text = f"{count} active {noun}."
        if self._selected_index is not None and self._selected_index < count:
            item = self._state_items[self._selected_index]
            text += (
                f" Selected {self._selected_index + 1}: feature {item.feature_id}, "
                f"strength {item.strength:g}, target {format_target(item)}."
            )
        summary.update(text)


def format_target(item: SteerItem) -> str:
    if item.sae_id:
        return item.sae_id
    if item.layers:
        return ",".join(str(layer) for layer in item.layers)
    return "-"


def copy_item_with_model_id(item: SteerItem, model_id: str) -> SteerItem:
    return SteerItem(
        feature_id=item.feature_id,
        strength=item.strength,
        layers=item.layers,
        label=item.label,
        model_id=model_id,
        sae_id=item.sae_id,
    )


def neuronpedia_sae_id_from_form(layers_raw: str, sae_raw: str) -> str:
    layers_raw = layers_raw.strip()
    sae_raw = sae_raw.strip()
    if layers_raw:
        return f"{parse_layers(layers_raw)[0]}-res-jb"

    if sae_raw:
        explicit_neuronpedia_sae = re.fullmatch(r"\d+-res[-_].+", sae_raw)
        if explicit_neuronpedia_sae:
            return sae_raw

        hook_layer = re.fullmatch(r"blocks\.(\d+)\.hook_resid_pre", sae_raw)
        if hook_layer:
            return f"{int(hook_layer.group(1))}-res-jb"

        raise SteeringError("lookup needs a layer or Neuronpedia SAE id")

    return "6-res-jb"


def feature_label_from_data(data: dict) -> str | None:
    explanations = data.get("explanations")
    if not isinstance(explanations, list):
        return None

    for explanation in explanations:
        if not isinstance(explanation, dict):
            continue
        description = explanation.get("description")
        if description is None:
            continue
        label = re.sub(r"\s+", " ", str(description)).strip()
        if label:
            return label
    return None


def steer_target_from_neuronpedia_source(source_id: str) -> tuple[str, str]:
    source_id = source_id.strip()
    layer_match = re.fullmatch(r"(\d+)-res-jb", source_id)
    if layer_match:
        return layer_match.group(1), ""
    return "", source_id


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def compact_label(text: str, *, max_chars: int = 80) -> str:
    label = compact_whitespace(text)
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 3].rstrip() + "..."


def format_cached_label(label: FeatureLabel) -> str:
    suffix = ""
    if label.explanation_model_name:
        suffix = f" [{label.explanation_model_name}]"
    return f"{label.model_id}/{label.source_id}/{label.feature_id}: {compact_whitespace(label.description)}{suffix}"


def format_lookup_strength(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:g}"
    try:
        return f"{float(str(value)):g}"
    except ValueError:
        return str(value)


def valid_token_count(raw: str) -> bool:
    try:
        value = int(raw.strip())
    except ValueError:
        return False
    return 1 <= value <= 512


def valid_temperature(raw: str) -> bool:
    try:
        value = float(raw.strip())
    except ValueError:
        return False
    return value >= 0


def completion_text_for_ui(text: str) -> tuple[str, bool]:
    stop_indices = [index for sequence in COMPLETION_STOP_SEQUENCES if (index := text.find(sequence)) >= 0]
    if not stop_indices:
        return text, False
    return text[: min(stop_indices)].rstrip(), True


def run_tui(
    *,
    server_url: str = "http://127.0.0.1:8000",
    max_tokens: int = DEFAULT_UI_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    state_path: Path | None = None,
) -> None:
    SteeringTUI(
        server_url=server_url,
        max_tokens=max_tokens,
        temperature=temperature,
        state_path=state_path or default_state_path(),
    ).run()


if __name__ == "__main__":
    run_tui()
