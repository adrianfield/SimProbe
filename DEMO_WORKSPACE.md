# SimProbe v0.4.5.2-final 演示工作区

产品定位为“仿真探索与边界分析”，一级页面为总览、场景资产、运行记录、评测分析和场景探索。

开发工作区保留完整历史运行、验收记录和测试夹具；演示工作区是只读复制形成的精简视图，两者相互隔离。

```powershell
python scripts/prepare_demo_workspace.py D:\demo\SimProbe
cd D:\demo\SimProbe
python -m streamlit run app.py
```

目标目录必须尚不存在。脚本只复制：

- 正式代码、依赖文件、README 和 `.env.example`；
- 正式 `data/sample_daily.xlsx` 与 `data/seed_scenarios.json`；
- 经脚本校验的真实 Qwen 高速汇入、前车制动运行记录 `CL-20260915-031`、`CL-20260915-032`；
- 对应成功批次 `BATCH-20260915-016`。

构建时会强制校验 Run 的 `llm_provider=qwen`、`llm_model=qwen3.8-flash`、`status=COMPLETED`、完整 `source_task_context`、非空局部状态转换事实与当前五阶段证据。脚本不会复制 `.env`、`.venv`、`test_*.py`、`acceptance_*.py`、缓存或测试专用数据；完成后还会执行确定性密钥扫描。它不会删除源文件、改写 Run 内容或篡改状态。
