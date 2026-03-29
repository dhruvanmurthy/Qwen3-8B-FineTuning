# Dataset Card: Qwen3-8B Tool Use Training Data

## Dataset Summary

Aggregated dataset of **40,000 curated examples** for training Qwen3-8B on multi-step tool use, function calling, and API orchestration.

### Dataset Composition

| Source | Count | Split | Purpose | Status |
|--------|-------|-------|---------|--------|
| **API-Bank** | 5,000 | 80/10/10 | Real API orchestration | ✅ Processed |
| **ToolBench** | 15,000 | 80/10/10 | Tool selection & chaining | ✅ Processed |
| **Gorilla** | 5,000 | 80/10/10 | Function calling accuracy | ✅ Processed |
| **Synthetic** | 15,000 | 80/10/10 | Edge cases & domain tasks | ✅ Generated |
| **Total** | **40,000** | **80/10/10** | Multi-domain tool use | ✅ Ready |

## Data Sources

### 1. API-Bank
- **Link**: https://huggingface.co/datasets/apibench/api-bank
- **Samples**: 5,000 (from ~50k available)
- **Focus**: Real-world API calls and sequences
- **Format**: Instruction-following with multi-step execution
- **License**: CC BY 4.0
- **Selection Criteria**: Diverse real-world APIs, max 3 API calls per sequence

**Example**:
```json
{
  "instruction": "Get the latest stock price for Apple and Microsoft, then calculate the average.",
  "tools": [
    {"name": "get_stock_price", "params": {"symbol": "AAPL"}},
    {"name": "get_stock_price", "params": {"symbol": "MSFT"}}
  ],
  "execution": [
    {"tool": "get_stock_price", "params": {"symbol": "AAPL"}, "result": 195.45},
    {"tool": "get_stock_price", "params": {"symbol": "MSFT"}, "result": 380.12}
  ],
  "output": "Apple: $195.45, Microsoft: $380.12. Average: $287.79"
}
```

### 2. ToolBench
- **Link**: https://huggingface.co/datasets/ToolBench/toolbench
- **Samples**: 15,000 (from ~200k available, balanced subset)
- **Focus**: Tool selection accuracy and chaining
- **Format**: Query + available tools + expected tool calls
- **License**: Apache 2.0
- **Selection Criteria**: Diverse tool categories, multi-step reasoning required

**Example**:
```json
{
  "query": "How can I send a email and create a calendar reminder for tomorrow?",
  "available_tools": [
    {"name": "send_email", "desc": "Send an email to recipient"},
    {"name": "create_reminder", "desc": "Create a calendar reminder"},
    {"name": "get_weather", "desc": "Get weather forecast"}
  ],
  "expected_calls": [
    {"tool": "send_email", "args": {"recipient": "user@example.com", "body": "Email body"}},
    {"tool": "create_reminder", "args": {"message": "Reminder", "time": "tomorrow"}}
  ]
}
```

### 3. Gorilla
- **Link**: https://huggingface.co/datasets/gorilla-llm/gorilla-dataset
- **Samples**: 5,000 (from ~10k available)
- **Focus**: Function calling with correct argument generation
- **Format**: API documentation + function calls
- **License**: MIT
- **Selection Criteria**: Wide API coverage, parameter diversity

**Example**:
```json
{
  "api_name": "stripe.payment_intents.create",
  "api_doc": "Creates a PaymentIntent object. After the PaymentIntent is created, you can attach a payment method and confirm to initiate the payment.",
  "functional_category": ["payment_processing"],
  "user_query": "Process a payment of $99.99 for customer cus_12345",
  "parameters": {
    "amount": 9999,
    "currency": "usd",
    "customer": "cus_12345"
  },
  "expected_call": "stripe.payment_intents.create(amount=9999, currency='usd', customer='cus_12345')"
}
```

### 4. Synthetic Data
- **Source**: Generated via GPT-4 and instruction templates
- **Samples**: 15,000
- **Focus**: Domain-specific scenarios, edge cases, error handling
- **Categories**:
  - E-commerce (3,000 samples)
  - Travel booking (3,000 samples)
  - Financial services (3,000 samples)
  - Content generation (3,000 samples)
  - Miscellaneous (3,000 samples)
- **Quality**: Human-reviewed (1,000 samples audited)

**Example Template**:
```json
{
  "category": "e-commerce",
  "instruction": "Help customer find products and process checkout",
  "tools": [
    {"name": "search_products", "params": ["query", "category", "price_range"]},
    {"name": "add_to_cart", "params": ["product_id", "quantity"]},
    {"name": "apply_coupon", "params": ["code"]},
    {"name": "checkout", "params": ["payment_method"]}
  ],
  "conversation": [
    {"role": "user", "content": "Find winter jackets under $200"},
    {"role": "assistant", "tool_calls": [{"tool": "search_products", "params": {"query": "winter jackets", "price_range": [0, 200]}}]},
    {"role": "user", "content": "Add the first result to cart"},
    {"role": "assistant", "tool_calls": [{"tool": "add_to_cart", "params": {"product_id": "prod_123", "quantity": 1}}]},
    {"role": "user", "content": "Apply code WINTER20"},
    {"role": "assistant", "tool_calls": [{"tool": "apply_coupon", "params": {"code": "WINTER20"}}]},
    {"role": "user", "content": "Proceed to checkout with credit card"},
    {"role": "assistant", "tool_calls": [{"tool": "checkout", "params": {"payment_method": "credit_card"}}]}
  ]
}
```

## Data Format

### Standardized Format (ChatML)
All datasets converted to unified chat format:

```json
{
  "id": "sample_12345",
  "source": "api-bank",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to the following tools..."
    },
    {
      "role": "user",
      "content": "Get the weather in San Francisco and New York",
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
              "type": "object",
              "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["C", "F"]}
              },
              "required": ["city"]
            }
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": "I'll get the weather for both cities.",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "get_weather", "arguments": "{\"city\": \"San Francisco\", \"units\": \"F\"}"}
        },
        {
          "id": "call_2",
          "type": "function",
          "function": {"name": "get_weather", "arguments": "{\"city\": \"New York\", \"units\": \"F\"}"}
        }
      ]
    },
    {
      "role": "user",
      "content": "[{\"id\": \"call_1\", \"result\": {\"temperature\": 72, \"condition\": \"sunny\"}}, {\"id\": \"call_2\", \"result\": {\"temperature\": 65, \"condition\": \"cloudy\"}}]"
    },
    {
      "role": "assistant",
      "content": "San Francisco is 72°F and sunny, while New York is 65°F and cloudy."
    }
  ]
}
```

## Statistics

### Data Distribution
```
Training Set:   32,000 samples (80%)
Validation Set:  4,000 samples (10%)
Test Set:        4,000 samples (10%)
```

### Tool Coverage
- **Unique Tools**: 487
- **Tool Categories**: 12 (API, fintech, e-commerce, etc.)
- **Max Tools per Example**: 5
- **Avg Tools per Example**: 2.3

### Sequence Length
```
Mean Input Length:    380 tokens
Median Input Length:  320 tokens
Max Input Length:     2048 tokens

Mean Output Length:   180 tokens
Median Output Length: 150 tokens
Max Output Length:    500 tokens
```

### Language
- **Primary**: English (97%)
- **Secondary**: Spanish, French, Chinese (3% mixed)

## Data Processing Pipeline

### Steps
1. **Download**: Fetch from HF Hub or API
2. **Clean**: Remove duplicates, invalid JSON, incomplete sequences
3. **Normalize**: Standardize tool call format
4. **Validate**: Schema validation, type checking
5. **Augment**: Add synthetic data
6. **Split**: 80/10/10 train/val/test
7. **Tokenize**: Prepare for training
8. **Upload**: Push to HF Hub as versioned dataset

### Quality Checks
✅ Schema validation (jsonschema)
✅ No duplicate examples (MD5 hash)
✅ Tool call executability (simulated)
✅ Language detection
✅ Token count bounds
✅ Balanced tool distribution

## Data Access

### Download
```bash
# Via Hugging Face Datasets library
from datasets import load_dataset

dataset = load_dataset("dhruvanmurthy/qwen3-8b-tool-use", split="train")
```

### Versioning
- **Latest**: v1.0 (March 2026)
- **Previous**: v0.9 (dev version)
- **DVC**: Tracked in `data/` directory with `.dvc` files

## Licensing & Attribution

| Source | License | Attribution |
|--------|---------|-------------|
| API-Bank | CC BY 4.0 | Xu et al., 2023 |
| ToolBench | Apache 2.0 | Qin et al., 2023 |
| Gorilla | MIT | Patil et al., 2023 |
| Synthetic | Custom (CC BY 4.0) | Generated, human-reviewed |

**Combined License**: CC BY 4.0 (most restrictive)

## Ethical Considerations

### Data Collection
- ✅ Datasets sourced from public, publicly-licensed sources
- ✅ No personally identifiable information (PII)
- ✅ No sensitive credentials in examples

### Potential Harms
1. **Synthetic Data Artifacts**: Generated examples may not reflect real user behavior
   - Mitigation: Balanced mix with real data (75%)
2. **API Bias**: Overrepresentation of popular APIs (Google, Stripe, AWS)
   - Mitigation: Intentional balancing across 487 unique tools
3. **Language Bias**: Primarily English; multilingual support limited
   - Mitigation: Flagged in model card; future work planned

### Bias Analysis
- ✅ Tool category distribution analyzed (no extreme skew)
- ✅ Geographic representation in examples (if applicable)
- ✅ Function complexity distribution balanced

## Data Versioning & Reproducibility

### Version Control
- **Git**: Code in GitHub
- **DVC**: Dataset snapshots in `data/` (`.dvc` files)
- **HF Hub**: Versioned dataset releases

### Reproducibility Checklist
✅ All preprocessing scripts in `scripts/prepare_datasets.sh`
✅ Random seeds fixed (42) for any sampling
✅ Exact dataset versions pinned
✅ License attribution preserved
✅ Processing configs versioned (YAML)

## Known Limitations

1. **Format Diversity**: Primarily ChatML format; OpenAI function calling influenced
2. **Domain Coverage**: Overweighted toward common APIs (e.g., weather, email)
3. **Error Cases**: Limited error handling examples (~5% of data)
4. **Multilingual**: Primarily English; non-English performance unknown
5. **Tool Coverage**: 487 tools may not generalize to all possible APIs

## Future Improvements

- [ ] Add more multilingual examples (Spanish, Mandarin, French)
- [ ] Expand error handling & edge cases
- [ ] Add streaming/real-time API examples
- [ ] Include authentication + credential handling patterns
- [ ] Cross-validate with additional tool-use benchmarks

## Contact

**Dataset Maintainer**: Dhruvan Murthy  
**HuggingFace**: [dhruvanmurthy/qwen3-8b-tool-use](https://huggingface.co/datasets/dhruvanmurthy/qwen3-8b-tool-use)  
**Issues**: GitHub issue tracker

---
**Last Updated**: March 2026  
**Version**: 1.0
