"""OpenAI-host research spikes (wire probes, refusal discovery, Arc 2 strict schema).

NOTE: this package is `spikes.openai`, NOT the vendor `openai` SDK. Nothing here
imports that SDK — trust boundary #8 confines it to `src/outrider/llm/`, and these
spikes reach the wire through `outrider.llm.raw_openai_capture`.
"""
