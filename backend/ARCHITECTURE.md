# LangGraph Multi-Agent Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Voice Call Center System                     │
│                      (LangGraph-based)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌────────────────────┐
                    │   User Input       │
                    │  "สวัสดีครับ..."    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Fast Path?       │
                    │  (Optimization)    │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   YES                 NO
                    │                   │
                    ▼                   ▼
        ┌────────────────────┐  ┌────────────────────┐
        │ Direct Response    │  │ State Graph        │
        │ (0.03s)            │  │ Processing         │
        └────────────────────┘  └─────────┬──────────┘
                                          │
                                          ▼
```

## State Graph Flow

```
                    ┌────────────────────────┐
                    │   SUPERVISOR NODE      │
                    │   Intent Classifier    │
                    │                        │
                    │ Input: user_input      │
                    │ Output: intent         │
                    │  - PRODUCT             │
                    │  - KNOWLEDGE           │
                    │  - GENERAL             │
                    └───────────┬────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
            PRODUCT │   KNOWLEDGE│   GENERAL│
                    │           │           │
                    ▼           ▼           ▼
        ┌───────────────┐ ┌────────────┐ ┌──────────────┐
        │ PRODUCT NODE  │ │ KNOWLEDGE  │ │ GENERAL NODE │
        │               │ │    NODE    │ │              │
        │ Tools:        │ │            │ │ Fast         │
        │ - Product DB  │ │ Tools:     │ │ Responses:   │
        │ - Sheets      │ │ - RAG      │ │ - Greetings  │
        │ - Time        │ │ - Time     │ │ - Thanks     │
        │               │ │            │ │ - Time       │
        └───────┬───────┘ └─────┬──────┘ └──────┬───────┘
                │               │                │
                └───────────────┼────────────────┘
                                │
                                ▼
                    ┌────────────────────┐
                    │   RESPONSE NODE    │
                    │   Composer         │
                    │                    │
                    │ Creates final      │
                    │ customer response  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   END              │
                    │   Return response  │
                    └────────────────────┘
```

## State Structure

```
┌────────────────────────────────────────────────────────────┐
│                    AgentState                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  INPUT:                                                    │
│  ├─ user_input: str              "สวัสดีครับ..."         │
│  └─ conversation_history: list   [{role, content}, ...]   │
│                                                            │
│  PROCESSING:                                               │
│  ├─ intent: str                  "PRODUCT"                │
│  ├─ intent_reason: str           "คำถามเกี่ยวกับสินค้า"   │
│  ├─ information: str             "พบสินค้า..."            │
│  └─ tool_outputs: list           ["product_db: ..."]      │
│                                                            │
│  OUTPUT:                                                   │
│  ├─ response: str                "สินค้า Galaxy A54..."  │
│  ├─ processing_start: float      1234567890.123          │
│  ├─ model_used: str              "qwen3-4b"               │
│  └─ route_taken: list            ["supervisor->product"]  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Tool Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Tools Layer                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Time Tool      │  │ Product DB     │  │ Sheets Tool  │ │
│  │                │  │ Tool           │  │              │ │
│  │ @tool          │  │                │  │ @tool        │ │
│  │ get_current    │  │ @tool          │  │ query_google │ │
│  │ _time()        │  │ query_product  │  │ _sheets()    │ │
│  │                │  │ _database()    │  │              │ │
│  │ Returns:       │  │                │  │ Returns:     │ │
│  │ Current time   │  │ Returns:       │  │ Sheet data   │ │
│  │ in Thailand    │  │ Product info   │  │              │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│                                                              │
│  ┌────────────────┐                                         │
│  │ RAG Tool       │                                         │
│  │                │                                         │
│  │ @tool          │                                         │
│  │ search_        │                                         │
│  │ documents()    │                                         │
│  │                │                                         │
│  │ Returns:       │                                         │
│  │ Doc excerpts   │                                         │
│  └────────────────┘                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Node Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Supervisor Node                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Receive state with user_input                          │
│  2. Build prompt with context                              │
│  3. Call LLM to classify intent                            │
│  4. Parse response (PRODUCT/KNOWLEDGE/GENERAL)             │
│  5. Update state.intent and state.route_taken              │
│  6. Return updated state                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────┐
│  Product Node   │ │ Knowledge   │ │ General     │
├─────────────────┤ │    Node     │ │   Node      │
│                 │ ├─────────────┤ ├─────────────┤
│ 1. Check cache  │ │ 1. Check    │ │ 1. Check    │
│ 2. Extract SKU  │ │    cache    │ │    fast     │
│ 3. Call tool:   │ │ 2. Call     │ │    patterns │
│    product_db() │ │    tool:    │ │ 2. Use LLM  │
│ 4. Or use LLM   │ │    search   │ │    if       │
│ 5. Update state │ │    docs()   │ │    needed   │
│    .information │ │ 3. Update   │ │ 3. Update   │
│ 6. Return state │ │    state    │ │    state    │
└─────────────────┘ └─────────────┘ └─────────────┘
            │             │             │
            └─────────────┼─────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Response Node                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Receive state with information                         │
│  2. Check if information is conversational                 │
│  3. If yes: use directly                                   │
│  4. If no: compose with LLM                                │
│  5. Update state.response                                  │
│  6. Return final state                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Routing Logic

```
                    ┌────────────────────┐
                    │ route_after_       │
                    │ supervisor()       │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Check state.intent │
                    └─────────┬──────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    if "PRODUCT"      if "KNOWLEDGE"    if "GENERAL"
            │                 │                 │
            ▼                 ▼                 ▼
    return "product"  return "knowledge" return "general"
            │                 │                 │
            └─────────────────┴─────────────────┘
                              │
                              ▼
                    Routes to appropriate node
```

## Fast Path Optimization

```
┌──────────────────────────────────────────────────────────┐
│                  Fast Path Logic                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Check user input for patterns:                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Pattern: Greeting                                  │ │
│  │ Keywords: สวัสดี, hello, hi, ดีครับ, ดีค่ะ        │ │
│  │ Response: "สวัสดีครับ ยินดีต้อนรับ..."             │ │
│  │ Time: ~0.03s                                       │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Pattern: Thanks                                    │ │
│  │ Keywords: ขอบคุณ, thank, ขอบใจ                    │ │
│  │ Response: "ยินดีครับ..."                           │ │
│  │ Time: ~0.03s                                       │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Pattern: Direct SKU                               │ │
│  │ Regex: (TEL|INT|TV)\d{3}                          │ │
│  │ Action: query_product_database(sku=match)         │ │
│  │ Time: ~0.05s                                       │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  If matched: Return immediately (bypass graph)          │
│  If not matched: Continue to state graph                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Caching Strategy

```
┌──────────────────────────────────────────────────────────┐
│                   Cache Layers                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Product Cache (OrderedDict, max 128 items)        │ │
│  │                                                    │ │
│  │ Key: f"product:{query_type}:{search_term}"        │ │
│  │ Value: Product information string                 │ │
│  │ Eviction: LRU (Least Recently Used)               │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ RAG Cache (OrderedDict, max 128 items)            │ │
│  │                                                    │ │
│  │ Key: f"rag:{query}|{top_k}"                        │ │
│  │ Value: Document excerpts                           │ │
│  │ Eviction: LRU                                      │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Sheets Cache (OrderedDict, max 128 items)         │ │
│  │                                                    │ │
│  │ Key: f"sheets:{name}|{op}|{query}|{data}"         │ │
│  │ Value: Sheets query result                         │ │
│  │ Eviction: LRU                                      │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
                    ┌────────────────┐
                    │ Try execute    │
                    │ node           │
                    └────────┬───────┘
                             │
                    ┌────────▼───────┐
                    │ Success?       │
                    └────────┬───────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                   YES               NO
                    │                 │
                    ▼                 ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ Continue to      │  │ Log error        │
        │ next node        │  │ with context     │
        └──────────────────┘  └─────────┬────────┘
                                        │
                                        ▼
                            ┌────────────────────┐
                            │ Fallback response  │
                            │ based on last      │
                            │ known state        │
                            └─────────┬──────────┘
                                      │
                                      ▼
                            ┌────────────────────┐
                            │ Return graceful    │
                            │ error message      │
                            └────────────────────┘
```

## Performance Profile

```
┌──────────────────────────────────────────────────────────┐
│              Response Time Breakdown                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Fast Path (Greeting):         0.031s                   │
│  ├─ Pattern matching:          0.001s                   │
│  └─ Response composition:      0.030s                   │
│                                                          │
│  Product Query (SKU):          0.520s                   │
│  ├─ Supervisor classification: 0.180s                   │
│  ├─ Product DB lookup:         0.050s                   │
│  ├─ Response composition:      0.290s                   │
│  └─ Overhead:                  0.000s                   │
│                                                          │
│  Knowledge Query (RAG):        0.950s                   │
│  ├─ Supervisor classification: 0.180s                   │
│  ├─ RAG search:                0.450s                   │
│  ├─ Response composition:      0.300s                   │
│  └─ Overhead:                  0.020s                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Comparison with CrewAI

```
┌──────────────────────────────────────────────────────────┐
│                    Architecture                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  CrewAI:                        LangGraph:               │
│                                                          │
│  Agent → Task → Agent           Node → State → Node     │
│  (Implicit flow)                (Explicit flow)          │
│                                                          │
│  ┌──────────┐                   ┌──────────┐            │
│  │  Agent   │                   │   Node   │            │
│  │          │                   │          │            │
│  │ Internal │                   │ Pure     │            │
│  │ state    │                   │ function │            │
│  │ (hidden) │                   │ (visible)│            │
│  └──────────┘                   └──────────┘            │
│                                                          │
│  Pros:                          Pros:                    │
│  • Simple abstraction           • Clear control          │
│  • Quick setup                  • Easy debugging         │
│                                 • Better performance     │
│                                 • Full observability     │
│                                                          │
│  Cons:                          Cons:                    │
│  • Black box                    • More setup code        │
│  • Hard to debug                • Manual routing         │
│  • Limited control                                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

**Visual Architecture Documentation**  
*Version 1.0 - November 24, 2025*
