"""Launch the deployed full attempt-extraction pipeline and return immediately."""

import modal


run_all = modal.Function.from_name("egoverse-attempt-extraction", "run_all")
call = run_all.spawn()
print(f"Started detached full extraction: {call.object_id}")
