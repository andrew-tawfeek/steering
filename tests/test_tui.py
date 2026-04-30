from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from textual.widgets import Button, DataTable, Input, Static

from steering.feature_cache import FeatureCache, FeatureLabel
from steering.state import SteerItem, load_state, update_state
from steering.tui import SteeringTUI, completion_text_for_ui, neuronpedia_sae_id_from_form


class FakeClient:
    base_url = "http://fake-server"

    def __init__(self) -> None:
        self.generations: list[dict] = []

    def health(self) -> dict:
        return {"model_name": "fake-model", "device": "cpu"}

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        yield "ok"


class FakeNeuronpedia:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def feature(self, model_id: str, sae_id: str, feature_id: int) -> dict:
        self.calls.append((model_id, sae_id, feature_id))
        return {
            "modelId": model_id,
            "layer": sae_id,
            "index": feature_id,
            "explanations": [{"description": " test   feature "}],
            "vectorDefaultSteerStrength": 6.5,
        }


class FakeDatasetClient:
    def __init__(self) -> None:
        self.sources = ["6-res-jb", "8-res-jb", "mlp-out"]
        self.downloads: list[tuple[str, str]] = []

    def list_sources(self, model_id: str) -> list[str]:
        return list(self.sources)

    def download_source_labels(self, model_id: str, source_id: str, *, max_files=None) -> list[FeatureLabel]:
        self.downloads.append((model_id, source_id))
        return [
            FeatureLabel(
                model_id,
                source_id,
                204,
                "time-related phrases and expressions",
                explanation_model_name="fake-explainer",
            ),
            FeatureLabel(model_id, source_id, 7, "grant funding references"),
        ]


class SlowFakeClient(FakeClient):
    def generate(self, **kwargs):
        self.generations.append(kwargs)
        yield "hel"
        time.sleep(0.15)
        yield "lo"


class ParagraphFakeClient(FakeClient):
    def generate(self, **kwargs):
        self.generations.append(kwargs)
        yield "first paragraph"
        yield "\n"
        yield "\n"
        yield "second paragraph"


class SteeringTUITests(unittest.IsolatedAsyncioTestCase):
    async def test_selecting_steer_populates_edit_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,), label="code"), append=False, path=path)
            update_state(SteerItem(7, -5, (8, 10)), append=True, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                self.assertEqual(app._selected_index, 0)
                self.assertEqual(app.query_one("#feature-id", Input).value, "204")

                app._select_steer(1)
                await pilot.pause(0.1)

                self.assertEqual(app.query_one("#feature-id", Input).value, "7")
                self.assertEqual(app.query_one("#strength", Input).value, "-5")
                self.assertEqual(app.query_one("#layers", Input).value, "8,10")

    async def test_save_edit_updates_selected_steer_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,)), append=False, path=path)
            update_state(SteerItem(7, -5, (8,)), append=True, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app._select_steer(1)
                app.query_one("#feature-id", Input).value = "9"
                app.query_one("#strength", Input).value = "3.5"
                app.query_one("#layers", Input).value = "5,6"
                app.query_one("#label", Input).value = "edited"

                app.update_selected_steer()
                await pilot.pause(0.1)

            state = load_state(path)
            self.assertEqual([item.feature_id for item in state.items], [204, 9])
            self.assertEqual(state.items[1].strength, 3.5)
            self.assertEqual(state.items[1].layers, (5, 6))
            self.assertEqual(state.items[1].label, "edited")

    async def test_enter_on_steer_table_opens_selected_row_for_editing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,), label="code"), append=False, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.action_focus_steers()

                await pilot.press("enter")
                await pilot.pause(0.1)

                self.assertEqual(app.focused.id, "feature-id")
                self.assertIn("Editing steer 1", str(app.query_one("#form-status", Static).content))

    async def test_enter_in_form_saves_current_steer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#feature-id", Input).value = "204"
                app.query_one("#strength", Input).value = "10"
                app.query_one("#layers", Input).value = "6"
                app.query_one("#feature-id", Input).focus()

                await pilot.press("enter")
                await pilot.pause(0.1)

            state = load_state(path)
            self.assertEqual(len(state.items), 1)
            self.assertEqual(state.items[0].feature_id, 204)
            self.assertEqual(state.items[0].strength, 10)
            self.assertEqual(state.items[0].layers, (6,))

    async def test_remove_selected_steer_keeps_remaining_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,)), append=False, path=path)
            update_state(SteerItem(7, -5, (8,)), append=True, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app._select_steer(0)
                app.remove_selected_steer()
                await pilot.pause(0.1)

                self.assertEqual(app._selected_index, 0)
                self.assertFalse(app.query_one("#remove-selected", Button).disabled)

            state = load_state(path)
            self.assertEqual(len(state.items), 1)
            self.assertEqual(state.items[0].feature_id, 7)

    async def test_invalid_generation_settings_do_not_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#prompt", Input).value = "hello"
                app.query_one("#max-tokens", Input).value = "0"

                app.send_prompt()
                await pilot.pause(0.1)

                self.assertFalse(app._generating)
                self.assertTrue(app.query_one("#max-tokens", Input).has_class("invalid"))

    async def test_negative_feature_id_marks_feature_field_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#feature-id", Input).value = "-1"
                app.query_one("#strength", Input).value = "10"
                app.query_one("#layers", Input).value = "6"
                await pilot.pause(0.1)

                app.save_steer(append=False)
                await pilot.pause(0.1)

                self.assertTrue(app.query_one("#feature-id", Input).has_class("invalid"))
                self.assertFalse(app.query_one("#layers", Input).has_class("invalid"))
                self.assertEqual(len(load_state(path).items), 0)

    async def test_generation_shows_stream_preview_while_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            app.client = SlowFakeClient()

            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#prompt", Input).value = "hello"

                app.send_prompt()
                await pilot.pause(0.08)

                preview = app.query_one("#stream-preview", Static)
                self.assertTrue(preview.display)
                self.assertIn("hel", app._stream_text)

                await pilot.pause(0.25)

                self.assertFalse(app._generating)
                self.assertFalse(preview.display)
                self.assertEqual(app._stream_text, "hello")

    async def test_generation_stops_at_first_paragraph_for_ui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            app.client = ParagraphFakeClient()

            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#prompt", Input).value = "hello"

                app.send_prompt()
                await pilot.pause(0.2)

                self.assertFalse(app._generating)
                self.assertEqual(app._stream_text, "first paragraph")
                self.assertIn("Stopped at first paragraph", str(app.query_one("#generation-status", Static).content))

    async def test_new_steer_clears_selection_and_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,), label="code"), append=False, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                app.action_new_steer()
                await pilot.pause(0.1)

                self.assertIsNone(app._selected_index)
                self.assertEqual(app.query_one("#feature-id", Input).value, "")
                self.assertEqual(app.query_one("#strength", Input).value, "")
                self.assertTrue(app.query_one("#edit-selected", Button).disabled)
                self.assertTrue(app.query_one("#remove-selected", Button).disabled)
                self.assertEqual(app.focused.id, "feature-id")

    async def test_clear_selection_clears_stale_edit_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,), label="code"), append=False, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                app.action_clear_selection()
                await pilot.pause(0.1)

                self.assertIsNone(app._selected_index)
                self.assertEqual(app.query_one("#feature-id", Input).value, "")
                self.assertEqual(app.query_one("#strength", Input).value, "")
                self.assertEqual(app.query_one("#layers", Input).value, "")
                self.assertTrue(app.query_one("#edit-selected", Button).disabled)
                self.assertEqual(app.focused.id, "feature-id")
                self.assertNotIn("Selected", str(app.query_one("#state-summary", Static).content))

    async def test_remove_last_steer_clears_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,), label="code"), append=False, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                app.remove_selected_steer()
                await pilot.pause(0.1)

                self.assertIsNone(app._selected_index)
                self.assertEqual(app.query_one("#feature-id", Input).value, "")
                self.assertEqual(app.query_one("#strength", Input).value, "")
                self.assertEqual(app.query_one("#layers", Input).value, "")
                self.assertTrue(app.query_one("#clear-steers", Button).disabled)

    async def test_clear_steers_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(SteerItem(204, 10, (6,)), append=False, path=path)
            update_state(SteerItem(7, -5, (8,)), append=True, path=path)

            app = make_app(path)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                app.clear_steers()
                await pilot.pause(0.1)

                self.assertEqual(len(load_state(path).items), 2)
                self.assertTrue(app._clear_confirm_pending)
                self.assertEqual(str(app.query_one("#clear-steers", Button).label), "Confirm Clear")

                app.clear_steers()
                await pilot.pause(0.1)

                self.assertEqual(len(load_state(path).items), 0)
                self.assertFalse(app._clear_confirm_pending)
                self.assertEqual(str(app.query_one("#clear-steers", Button).label), "Clear All")
                self.assertTrue(app.query_one("#clear-steers", Button).disabled)

    async def test_lookup_infers_neuronpedia_sae_from_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            fake_neuronpedia = FakeNeuronpedia()
            app.neuronpedia = fake_neuronpedia

            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#feature-id", Input).value = "204"
                app.query_one("#layers", Input).value = ""
                app.query_one("#sae-id", Input).value = "blocks.8.hook_resid_pre"

                app.lookup_feature()
                self.assertTrue(app.query_one("#lookup", Button).disabled)
                await pilot.pause(0.2)

                self.assertEqual(fake_neuronpedia.calls, [("gpt2-small", "8-res-jb", 204)])
                self.assertFalse(app._lookup_running)
                self.assertFalse(app.query_one("#lookup", Button).disabled)
                self.assertEqual(app.query_one("#label", Input).value, "test feature")
                self.assertEqual(app.query_one("#strength", Input).value, "6.5")

    async def test_lookup_does_not_overwrite_existing_label_or_strength(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            fake_neuronpedia = FakeNeuronpedia()
            app.neuronpedia = fake_neuronpedia

            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#feature-id", Input).value = "204"
                app.query_one("#layers", Input).value = "8"
                app.query_one("#label", Input).value = "keep me"
                app.query_one("#strength", Input).value = "3"

                app.lookup_feature()
                await pilot.pause(0.2)

                self.assertEqual(app.query_one("#label", Input).value, "keep me")
                self.assertEqual(app.query_one("#strength", Input).value, "3")

    async def test_backend_unavailable_blocks_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)

            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                app._backend_available = False
                app._sync_buttons()
                app.query_one("#prompt", Input).value = "hello"

                app.send_prompt()
                await pilot.pause(0.1)

                client = app.client
                self.assertIsInstance(client, FakeClient)
                self.assertEqual(client.generations, [])
                self.assertEqual(app.query_one("#prompt", Input).value, "hello")
                self.assertTrue(app.query_one("#send", Button).disabled)

    async def test_feature_cache_lists_sources_and_selects_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            app.dataset_client = FakeDatasetClient()

            async with app.run_test(size=(140, 48)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#cache-model-id", Input).value = "gpt2-small"
                app.query_one("#cache-source-filter", Input).value = "res"

                app.list_cache_sources()
                await pilot.pause(0.2)

                self.assertEqual(app._cache_sources, ("6-res-jb", "8-res-jb"))
                self.assertEqual(app.query_one("#source-table", DataTable).row_count, 2)
                app._select_cache_source(1)

                self.assertEqual(app.query_one("#cache-source", Input).value, "8-res-jb")

    async def test_feature_cache_download_search_and_apply_to_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            fake_dataset = FakeDatasetClient()
            app.dataset_client = fake_dataset

            async with app.run_test(size=(140, 48)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#cache-model-id", Input).value = "gpt2-small"
                app.query_one("#cache-source", Input).value = "6-res-jb"

                app.download_cache_source()
                await pilot.pause(0.2)

                self.assertEqual(fake_dataset.downloads, [("gpt2-small", "6-res-jb")])
                self.assertEqual(FeatureCache(app.cache_path).status()[0].label_count, 2)

                app.query_one("#cache-query", Input).value = "time phrases"
                app.search_feature_cache()
                await pilot.pause(0.2)

                self.assertEqual([label.feature_id for label in app._cache_results], [204])
                app.apply_cached_feature()
                await pilot.pause(0.1)

                self.assertEqual(app.query_one("#feature-id", Input).value, "204")
                self.assertEqual(app.query_one("#layers", Input).value, "6")
                self.assertEqual(app.query_one("#sae-id", Input).value, "")
                self.assertEqual(app.query_one("#model-id", Input).value, "gpt2-small")
                self.assertEqual(app.query_one("#label", Input).value, "time-related phrases and expressions")
                self.assertEqual(app.query_one("#strength", Input).value, "10")

    async def test_feature_cache_inspects_feature_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            app = make_app(path)
            FeatureCache(app.cache_path).replace_source(
                "gpt2-small",
                "6-res-jb",
                [
                    FeatureLabel("gpt2-small", "6-res-jb", 204, "time-related phrases"),
                    FeatureLabel("gpt2-small", "6-res-jb", 7, "grant funding references"),
                ],
            )

            async with app.run_test(size=(140, 48)) as pilot:
                await pilot.pause(0.1)
                app.query_one("#cache-model-id", Input).value = "gpt2-small"
                app.query_one("#cache-source", Input).value = "6-res-jb"
                app.query_one("#cache-feature-id", Input).value = "7"

                app.inspect_cached_feature()
                await pilot.pause(0.2)

                self.assertEqual(len(app._cache_results), 1)
                self.assertEqual(app._cache_results[0].description, "grant funding references")

    def test_neuronpedia_sae_id_from_form_prefers_layers(self) -> None:
        self.assertEqual(neuronpedia_sae_id_from_form("8,10", "blocks.6.hook_resid_pre"), "8-res-jb")
        self.assertEqual(neuronpedia_sae_id_from_form("", "blocks.6.hook_resid_pre"), "6-res-jb")
        self.assertEqual(neuronpedia_sae_id_from_form("", "7-res-jb"), "7-res-jb")
        self.assertEqual(neuronpedia_sae_id_from_form("", "6-res_scefr-ajt"), "6-res_scefr-ajt")

    def test_completion_text_for_ui_trims_after_first_paragraph(self) -> None:
        self.assertEqual(completion_text_for_ui("first\n\nsecond"), ("first", True))
        self.assertEqual(completion_text_for_ui("first\r\n\r\nsecond"), ("first", True))
        self.assertEqual(completion_text_for_ui("one line"), ("one line", False))


def make_app(path: Path) -> SteeringTUI:
    app = SteeringTUI(
        server_url="http://127.0.0.1:8000",
        max_tokens=1,
        temperature=0,
        state_path=path,
        cache_path=path.with_name("features.sqlite3"),
    )
    app.client = FakeClient()
    return app


if __name__ == "__main__":
    unittest.main()
