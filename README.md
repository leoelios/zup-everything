# TODO

- [x] Multi agent parallelization
- [x] Use stream version of chat api
- [x] Overwrite mode vs edit mode
- [x] Incluir tool de consulta web
- [ ] existem algumas informacoes que o agente pode julgar importante para ser utilizada no futuro, colocar essas informacoes no knowledge source com slug ever-test-01 e

## agent configuration

```json
{
  "knowledge_sources_details": [
    {
      "id": "01KKYH2Q57H53KKKD63G14Y9BV",
      "slug": "ever-test-01",
      "name": "everything-test-01",
      "description": "",
      "type": "custom",
      "creator": "Leonardo Elias De Oliveira",
      "default": false,
      "visibility_level": "personal",
      "model_name": "text-embedding-3-large"
    }
  ],
  "knowledge_sources_config": {
    "knowledge_sources": ["01KKYH2Q57H53KKKD63G14Y9BV"],
    "max_number_of_kos": 4,
    "relevancy_threshold": 40,
    "similarity_function": "COSINE",
    "post_processing": false,
    "knowledge_sources_details": [
      {
        "id": "01KKYH2Q57H53KKKD63G14Y9BV",
        "slug": "ever-test-01",
        "name": "everything-test-01",
        "description": "",
        "type": "custom",
        "creator": "Leonardo Elias De Oliveira",
        "default": false,
        "visibility_level": "personal",
        "model_name": "text-embedding-3-large"
      }
    ],
    "sealed": false
  },
  "type": "CONVERSATIONAL",
  "avatar": "https://genai-images.stackspot.com/01GRVXMNV1P0281NGWBX71D752/agents/01KKYST8HV5X3G36Q5M3NBQ0K0.jpg?timestamp=1773781459",
  "suggested_prompts": [],
  "system_prompt": "You are Zup CLI, an AI coding assistant. You help users with software engineering \\\ntasks by reading and writing files, executing shell commands, and managing knowledge sources.",
  "name": "zup-cli-agent",
  "slug": "46e2f0-zup-cli-agent",
  "llm_settings": {
    "reasoning_effort": "high",
    "verbosity": "high"
  },
  "model_id": "01KA9AA6HGFPJQ3SS638R3V7RY",
  "model_name": "gpt-5.1",
  "memory": "buffer",
  "planner_type": "simple",
  "max_llm_interactions": 200,
  "mode": "autonomous",
  "structured_output": null,
  "tools": [],
  "mcpToolkits": [],
  "sub_agents_ids": [],
  "enabled_structured_outputs": true,
  "enabled_tools": true,
  "available_llm_models": [
    {
      "model_id": "01KA9AA6HGFPJQ3SS638R3V7RY",
      "model_name": "GPT 5.1",
      "is_default": true
    }
  ],
  "builtin_tools_ids": [],
  "custom_tools": [],
  "mcp_toolkit_ids": [],
  "available_models_ids": ["01KA9AA6HGFPJQ3SS638R3V7RY"],
  "status": "published"
}
```
