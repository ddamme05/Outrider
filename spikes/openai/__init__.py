"""Research tools for OpenAI request formats, refusals and strict schemas.

spikes.openai is this project's experiment package, separate from the vendor
openai SDK. Calls go through outrider.llm.raw_openai_capture; trust boundary #8
keeps SDK use in the application wrapper.
"""
