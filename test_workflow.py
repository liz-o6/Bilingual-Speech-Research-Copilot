import asyncio
import os
import json
from pathlib import Path
from datetime import datetime

from src.agents.research import research as research_agent
from src.agents.data import data as data_agent
from src.agents.summary import summary as summary_agent

from src.orchestration.state import State
from src.orchestration.supervisor import Supervisor
from src.orchestration.graph import build_graph, run_workflow


async def main():
    question = ""
    folder = None
    events = []
    traces = []

    def save():
        # Save after each node so completed steps survive interruptions.
        report = {"question": question, "events": events, "traces": traces}
        (folder / "workflow.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# Workflow debug", question]
        for event in events:
            lines.append("\n## " + event["event"])
            lines.append("```json\n" + json.dumps(event, ensure_ascii=False, indent=2) + "\n```")
        lines.append("\n## 逐次调用、实际证据和引用核验")
        for trace in traces:
            lines.append("```json\n" + json.dumps(trace, ensure_ascii=False, indent=2) + "\n```")
        (folder / "workflow.md").write_text("\n".join(lines), encoding="utf-8")

    supervisor = Supervisor(traces=traces)
    print("Anthropic 模型：", os.getenv("ANTHROPIC_MODEL", "未配置"), "；thinking 已关闭")

    async def debug_supervisor(current):
        events.append({"event": "supervisor_input", "state": current.model_dump(mode="json")})
        save()
        updates = await supervisor(current)
        # Convert updates through State to serialize DocumentRequest values.
        updated = current.model_copy(update=updates)
        events.append({"event": "supervisor_output", "state": updated.model_dump(mode="json")})
        save()
        return updates

    async def debug_worker(agent, name, current):
        events.append({"event": name + "_input", "state": current.model_dump(mode="json")})
        save()
        try:
            output = await agent(current, traces=traces)
            events.append({"event": name + "_output", "result": output.model_dump(mode="json")})
            return output
        except Exception as exc:
            events.append({"event": name + "_error", "error_type": type(exc).__name__})
            raise
        finally:
            save()

    async def debug_research(current):
        return await debug_worker(research_agent, "research", current)

    async def debug_data(current):
        return await debug_worker(data_agent, "data", current)

    async def debug_summary(current):
        return await debug_worker(summary_agent, "summary", current)

    graph = build_graph(
        supervisor=debug_supervisor, research_agent=debug_research,
        data_agent=debug_data, summary_agent=debug_summary,
    )

    while True:
        try:
            question = input("\n请输入测试问题（输入 exit 退出）：").strip()
        except EOFError:
            break
        if question.lower() == "exit":
            break
        if not question:
            continue

        # Clear in place; the Supervisor callback shares this list.
        events.clear()
        traces.clear()
        folder = Path(__file__).parent / "artifacts" / "debug" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        folder.mkdir(parents=True)
        state = State(query=question, research_task="", data_task="", reason="")
        print("Debug 文件夹：", folder)
        save()
        try:
            result = await run_workflow(graph, state)
        except Exception as exc:
            # Keep logs after failures and continue; hide raw errors that may contain keys.
            events.append({"event": "workflow_error", "error_type": type(exc).__name__})
            print("本次执行失败：", type(exc).__name__)
        else:
            print_result(result)
        finally:
            save()
            calls = [t for t in traces if t.get("event") == "llm_call"]
            print("本题模型调用次数：", len(calls))
            for call in calls:
                usage = call.get("usage") or {}
                print(call["agent"], call["model"], call["status"],
                      "输入：", usage.get("input_tokens", "未知"),
                      "输出：", usage.get("output_tokens", "未知"))

    print("已退出。")


def print_result(result):
    print("\n===== 最终回答 =====")
    print(result.final_answer)

    print("\n===== Supervisor =====")
    print("初始 Agent 计划：", result.agent_plan)
    print("修正次数：", result.repair_count)
    print("结束状态：", result.next_agent)
    print("执行次数：", result.step_count)
    print("最后决策理由：", result.reason)

    if result.research_result is not None:
        research = result.research_result
        print("\n===== Research =====")
        print("状态：", research.status)
        print("错误：", research.error)
        print("来源编号：", list(research.sources))
        print("\n分配给 Research 的任务：", result.research_task)
        print("实际检索词：", result.research_query or result.query)
        print("是否启用联网：", result.research_use_web)
        print("\n===== Research 完整回答 =====")
        print(research.answer)

        print("\n===== Research 来源详情 =====")
        for label, source in research.sources.items():
            print(f"\n--- [{label}] ---")
            print("论文 ID：", source.get("doc_id"))
            print("标题：", source.get("title"))
            print("页码：", source.get("page"))
            print("章节：", source.get("section_id"))
            print("片段 ID：", source.get("chunk_id"))
            print("检索距离：", source.get("distance"))
            print("片段全文：")
            print(source.get("text", ""))

    if result.data_result is not None:
        data = result.data_result
        print("\n===== Data =====")
        print("状态：", data.status)
        print("错误：", data.error)
        print("实际计算结果：", data.rows)

    if result.summary_result is not None:
        summary = result.summary_result
        print("\n===== Summary =====")
        print("状态：", summary.status)
        print("错误：", summary.error)
        print("供应商：", summary.provider)
        print("路由理由：", summary.route_reason)
        print("引用检查：", summary.citation_checks)


if __name__ == "__main__":
    asyncio.run(main())
