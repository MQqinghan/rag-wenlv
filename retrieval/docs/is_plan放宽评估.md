# is_plan 放宽评估（④，扩大铁路/POI/行程工具覆盖）

> 2026-09-10 | 起因：T14 备注「铁路仅覆盖 `is_plan=True` 的跨城行程，而 v2 路由把『我想去云南玩 5 天…怎么安排』判为 `is_plan=False`」
> 方法：先逐行定位代码 → 实测 10 条问句的 v2 原始判定 → 量化 50 题影响面 → 离线 A/B 三变体
> 状态：**方案 C 已实施并验证通过（2026-09-10，主人拍板）** —— 见第八节「实施记录」

---

## 一、问题定位（代码级）

| 位置 | 事实 |
|---|---|
| `intent_route_service.py:83-87` | `_PLAN_QUERY_PATTERN` 只收「规划/帮我安排/安排一下/行程安排/路线安排/旅游计划/\d+天\d*晚/怎么玩」等**显式**句式 |
| `intent_route_service.py:636-649` | `classify_intent_v2` 走**强信号短路**：命中 `_PLAN_QUERY_PATTERN` 直接 `is_plan=True` 并返回 |
| `intent_route_service.py:628-629` | 未短路时才由 v2 LLM 决定 `is_plan`（`_apply_route_v2`） |
| `intent_route_v2.prompt:22` | is_plan 规则：「true 当用户在做『行程/路线/日程/几天怎么玩/怎么安排』的可执行规划请求……纯知识问答 is_plan 仍为 false」 |
| `intent_route_v2.prompt:32-41` | 末尾**唯一示例 JSON** 写的是 `"is_plan": false, "tools": ["none"]` |

即：`is_plan` 有两条来源——**正则强信号**（确定性）与 **v2 LLM**（概率性）。

## 二、实测：v2 LLM 的 is_plan 恒为 false（10/10）

直接调 `_call_route_v2_model` 看**原始 JSON**（绕过一切缓存）：

| 问句 | v2.is_plan | tools | 模型理由（原文节选） |
|---|---|---|---|
| 帮我规划一下去西安的行程 | **False** | none | 「但未明确具体需求，属于初步询问」 |
| 成都三天怎么玩 | **False** | none | 「属于旅游域的知识型问题，但不是明确的行程规划请求」 |
| 我想去云南玩 5 天，昆明、大理、丽江怎么安排比较顺？ | **False** | none | 「但未明确需要具体路线工具」 |
| 下周一从深圳出发去成都玩三天，行程怎么安排比较好？ | **False** | none | — |
| 大理丽江 6 天自由行路线怎么走 | **False** | none | — |
| 杭州三日游攻略 / 国庆想去厦门玩帮我安排一下 / 带父母去北京玩 4 天怎么安排合理 | **False** | none | — |
| 成都市区有什么好玩的景点（知识型，应 False） | False | none | ✅ 正确 |

**结论**：v2 LLM 对**任何**问句都判 `is_plan=false`，`tools` 恒为 `["none"]`——设计上应由 LLM 主导的判断**完全失效**，`is_plan=True` 实际只由正则产生。

## 三、影响面：50 题里只有 3 条命中正则

| 项 | 结果 |
|---|---|
| 现正则命中（=is_plan True） | **3 条**：`plan-001`、`neg-003`、`neg-006` |
| 金标为 plan 但**未进行程分支** | **4 条**：`plan-002`、`plan-003`、`plan-004`、`plan-005` |
| 扩正则会误伤的 | **1 条**：`tr-008`「第一次去杭州，景点应该怎么安排？」（金标 answer） |

> ⚠️ 这意味着 **5 条金标 plan 里 4 条其实没走行程链路**（铁路/天气/路线/POI 全部缺席），
> 而评测 Acc 仍是 1.0——**judge 只看答案是否覆盖金标要点，检不出"该生成行程却走了普通问答"**。
> 这是评测盲区，单靠全量 Acc 无法发现。

## 四、离线 A/B：根因是「示例 JSON 锚定」

在 `logs/_tmp_is_plan_probe.py` 中构造三个变体，对 10 条问句（5 条金标 plan + 5 条负例）直接调 LLM 比对：

| 变体 | 做法 | 命中期望 |
|---|---|---|
| V0 现状 | 原文 prompt | 5/10 |
| V1 只改规则 | 把 is_plan/tools 规则改成"命中任一条即 true"的硬标准，**示例不动** | **5/10（毫无变化）** |
| V2 规则 + 示例 | V1 基础上，把末尾示例改成「行程请求 is_plan=true / 知识咨询 false」**两个正反例** | **8/10** |

V2 结果：5 条金标 plan **全部** `is_plan=true` 且 `tools` 正确带 `route`（部分带 `weather`/`poi`）；
负例「成都有什么好吃的」「云南什么季节去合适」「明天天气」均正确 False。

**根因确认**：prompt 末尾**唯一示例**里的 `"is_plan": false, "tools": ["none"]` 对模型形成**强锚定**——
只改文字规则无效，必须同时修正示例（这是本次最有价值的发现，也可推广到其它 JSON 输出 prompt）。

**V2 仍未解决的 2 例**（过度触发）：

| 问句 | 期望 | V2 | 说明 |
|---|---|---|---|
| tr-008「第一次去杭州，景点应该怎么安排？」 | False | True | 含"怎么安排"但无天数/出发地，属**咨询** |
| 「…撒哈拉沙漠怎么去？值得去吗？」 | False | True | T7 已将"值得去吗"从正则移除，但 LLM 仍误判 |

## 五、方案对比

| 方案 | 做法 | 修好 plan-002~005 | 误伤 tr-008/撒哈拉 | 风险 |
|---|---|---|---|---|
| **A 只改规则** | 改 prompt 规则文字 | ❌ 无效（实测 0 改善） | — | 无收益，不采纳 |
| **B 改规则+示例** | V2 变体 | ✅ 5/5 | ⚠️ 2 条误伤 | 中：可能把咨询类变成行程生成 |
| **C B + 确定性降级守卫（推荐）** | B 之后，在 `_apply_route_v2` 加守卫：`is_plan=True` 但问句**不含任何行程定义词**（天数含中文数词 / 行程 / 规划 / 路线 / 自由行 / 几日游 / 怎么玩 / 接下来 / 到了…之后）→ 降回 False | ✅ 5/5（4 条含"天数"或"接下来"） | ✅ 0（tr-008 无天数无"接下来"、撒哈拉无行程词 → 降回 False） | 低：LLM 负责"宽松识别"，守卫负责"硬性收敛" |
| **D 不动，仅扩正则** | 扩 `_PLAN_QUERY_PATTERN` | ✅ 4/5（plan-004"玩两三天"需补中文数词） | ⚠️ tr-008 仍误伤（"怎么安排"） | 中：正则越扩越脆，且未修 LLM 失效的根因 |
| **E 全部不动** | 保持现状 | ❌ | — | 铁路/POI 覆盖面长期受限 |

## 六、推荐：方案 C

**理由**：
1. **B 修的是根因**（示例锚定导致 LLM 完全失效），收益不止于 is_plan——`tools`/`source_policy` 的判断也随之恢复（实测 tools 已正确带 route/weather/poi）；
2. **守卫提供确定性收敛**，把"咨询类"挡在行程分支外，正好补 B 的 2 处误伤，且规则可枚举、可单测；
3. 与 T7 的教训一致：**LLM 判定负责"宽召回"，确定性代码负责"严收敛"**——与本项目"prompt 约束遵循度不足则代码级兜底"的一贯做法相同。

**改动清单（待拍板后执行）**：

| 文件 | 改动 |
|---|---|
| `app/resources/prompts/tourism/intent_route_v2.prompt` | ① is_plan 规则改硬标准（三类命中即 true + 明示"缺出发地/日期不得判 false"）；② tools 规则加"is_plan=true 时必含 route"；③ **末尾示例改为"行程请求 true" + "知识咨询 false" 两个正反例** |
| `app/rag/common/intent_route_service.py` | `_apply_route_v2` 增加 `_PLAN_SIGNAL_PATTERN` 守卫（含中文数词），`is_plan=True` 但无信号 → 降回 False 并打日志 |
| `test/test_intent_route_v2.py` | 补守卫单测（tr-008 类降级、plan-002~005 类保留） |

**护栏（不可违反）**：
- `plan-001~005` 必须全部 `is_plan=True`（**当前 4 条不达标，属修复目标**）；
- `tr-008`、`xr-001`（撒哈拉）、`cl-002` 等咨询/澄清类必须保持 `is_plan=False`；
- 全量 50 题 **Acc 1.0 不破**（重点看 plan 分域与 tr-* ）。

## 七、附：评测盲区（独立于本方案，建议补）

现有 judge 只判"答案是否覆盖金标要点"，**检不出路由形态错误**（该出行程却出普通问答仍能拿满分）。
建议后续在 `eval_cases.json` 的 plan 类用例上增加**结构断言**（如答案含"行程概览/每日安排/花销预估"或 `is_plan=True`），
与 trip 评测已有的 `_judge_structure` 同思路。

---

## 八、实施记录（2026-09-10，方案 C 已落地）

### 8.1 改动

| 文件 | 改动 |
|---|---|
| `app/resources/prompts/tourism/intent_route_v2.prompt` | ① is_plan 规则改硬标准（三类命中即 true + 明示「缺出发地/日期/预算不得判 false」）；② tools 加「is_plan=true 必含 route，只有 is_plan=false 才允许 none」；③ 末尾示例由**单个** `is_plan:false` 改为**两例正反例**（示例A 行程请求 true / 示例B 知识咨询 false） |
| `app/rag/common/intent_route_service.py` | 新增 `_PLAN_SIGNAL_PATTERN`（行程定义词，**刻意不含「怎么安排」**，否则 tr-008 类失效）+ `_apply_route_v2` 守卫：LLM 判 `is_plan=True` 但问句无信号 → 降回 False，并去掉 `tools.route`；路由日志改打**生效值**（附 LLM 原始值） |
| `app/rag/common/intent_route_service.py` | **附带修一个会掩盖本次修复的坑**：v2 语义缓存键只含 `(问题, 历史)`、**不含 prompt 内容** → 改 prompt 后旧缓存仍会被命中。新增 `_ROUTE_PROMPT_REV = "r2"` 并入 `cache_parts`；**约定：此后改 `intent_route_v2.prompt` 必须递增该标记** |
| `test/test_intent_route_v2.py` | 新增第 8 节守卫单测（7 条：4 条应保留 / 3 条应降回，并断言降回后 tools 无 route） |

### 8.2 验证

| 项 | 结果 |
|---|---|
| 真实 v2 路由（新 prompt，绕开旧缓存） | `plan-001~005` **5/5 `is_plan=True`**（修前仅 plan-001，4 条不达标）；`tr-008`/`xr-001`/`tr-005`/`cl-003` 被守卫正确降回 False |
| 单测 | `test/test_intent_route_v2.py` **8/8**（含新增守卫 7/7）；L0 state 键扫描（27 键）通过 |
| 定向评测（20 条：plan5 + tr/xr/cl/neg/ab） | **Acc 1.0 / P 1.0 / R 1.0 / F1 1.0，FP 0 / FN 0**（`output/eval/20260910_isplan_c`） |
| 行程链路取证（日志） | `node_tool_rail` / `node_tool_poi` / `node_itinerary_generate` **各执行 8 次**（= plan-001~005 + tr-009 + neg-003 + neg-006），**修前仅 3 条** |
| 车次真实性抽查 | plan-003 答案中的 `Z8006` 经 `rail_gateway` 核为**真实车次**（深圳东 18:02 → 三亚 13:10，直特 19h8m，硬座 267.5 元），白名单机制正常、无编造 |

### 8.3 遗留观察项（已记录，未改）

- **`tr-009`「川西小环线5天自驾怎么走？」现也进行程分支**（含"5天"，LLM 判 true 且守卫保留）。金标 `expect_type=answer`，本轮 judge 仍 1.0——属**行为变化**而非回归；是否需要把"自驾路线咨询"排除在行程生成外，待主人判断（若要求，可在 `_PLAN_SIGNAL_PATTERN` 侧细化）。
- **第七节的评测盲区仍未补**：judge 检不出「该出行程却走普通问答」。本轮靠"节点执行次数"日志人工取证，建议后续为 plan 类用例补结构断言。
- **T14 收益仍未完全兑现**：行程答案的"交通方式/花销预估"仍写"以 12306 为准"，未主动消费【铁路参考】中的真实票价（见 T14 遗留观察）。

---

**取证脚本**：`logs/_tmp_is_plan_probe.py`（A/B 三变体，logs/ 已 gitignore）。

