import unittest

from ScratchPad_Agent import Scratchpad, ScratchpadAgent


class ScratchpadTest(unittest.TestCase):
    def test_write_and_read_value(self) -> None:
        scratchpad = Scratchpad()

        scratchpad.write("profit_margin", 0.42, "Product margin")

        self.assertEqual(scratchpad.read("profit_margin"), 0.42)
        self.assertEqual(scratchpad.list_keys(), ["profit_margin"])

    def test_clear_removes_notes(self) -> None:
        scratchpad = Scratchpad()
        scratchpad.write("temporary_result", {"value": 1})

        scratchpad.clear()

        self.assertIsNone(scratchpad.read("temporary_result"))
        self.assertEqual(scratchpad.list_keys(), [])

    def test_prompt_text_formats_structured_memory(self) -> None:
        scratchpad = Scratchpad()
        scratchpad.write("fastest_growth", "Product B", "Growth leader")

        prompt_text = scratchpad.to_prompt_text()

        self.assertIn("[Working Memory - Known Information]", prompt_text)
        self.assertIn("fastest_growth (Growth leader)", prompt_text)
        self.assertIn('"Product B"', prompt_text)


class ScratchpadAgentToolTest(unittest.TestCase):
    def test_save_tool_writes_to_memory(self) -> None:
        agent = ScratchpadAgent(client=object())

        result = agent._execute_tool(
            "save_to_scratchpad",
            {"key": "total_revenue", "value": 465, "description": "Q1 total"},
        )

        self.assertIn("Saved: total_revenue", result)
        self.assertEqual(agent.scratchpad.read("total_revenue"), 465)

    def test_read_tool_returns_saved_value(self) -> None:
        agent = ScratchpadAgent(client=object())
        agent.scratchpad.write("low_margin_product", "Product D")

        result = agent._execute_tool(
            "read_from_scratchpad",
            {"key": "low_margin_product"},
        )

        self.assertEqual(result, 'low_margin_product = "Product D"')

    def test_list_tool_returns_memory_keys(self) -> None:
        agent = ScratchpadAgent(client=object())
        agent.scratchpad.write("profit_margin", 0.42)

        result = agent._execute_tool("list_scratchpad_keys", {})

        self.assertIn("profit_margin", result)

    def test_tool_execution_is_recorded_in_trace(self) -> None:
        agent = ScratchpadAgent(client=object())

        agent._execute_tool(
            "save_to_scratchpad",
            {"key": "total_revenue", "value": 465},
        )

        trace = agent.get_trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["type"], "tool_execution")
        self.assertEqual(trace[0]["tool_name"], "save_to_scratchpad")
        self.assertEqual(trace[0]["memory_snapshot"]["total_revenue"]["value"], 465)

    def test_trace_returns_a_copy(self) -> None:
        agent = ScratchpadAgent(client=object())
        agent._execute_tool("list_scratchpad_keys", {})

        trace = agent.get_trace()
        trace[0]["type"] = "mutated"

        self.assertEqual(agent.get_trace()[0]["type"], "tool_execution")


if __name__ == "__main__":
    unittest.main()
