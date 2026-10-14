"""
Headless validation script and automated benchmark for Mesa Self-Construction.
Runs all experiments to completion, verifies pattern invariants, and outputs a validation summary.
"""

import time
from mesa_self_construction import SelfConstructionModel, ALL_PRESETS


def run_all_experiments():
    print("=" * 70)
    print("  MESA SELF-CONSTRUCTION: AUTOMATED EXPERIMENT VALIDATION SUITE")
    print("=" * 70)

    results = []

    for preset_id, preset in ALL_PRESETS.items():
        print(f"\n[Running Experiment] {preset.name} ({preset_id})")
        print(f"  Category: {preset.category}")
        print(f"  Target validation step: {preset.validation_step}")

        model = SelfConstructionModel(preset_id=preset_id)
        start_time = time.perf_counter()

        # Step through until validation step
        for _ in range(preset.validation_step):
            model.step()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        val = model.latest_validation

        # Strict: only exact box match + population + settling/constructions.
        status = "PASSED" if val.passed else "FAILED"
        h, w = preset.grid_size
        cell_updates = h * w * preset.validation_step
        mus_per_s = (cell_updates / elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
        extra = ""
        if model.is_von_neumann:
            extra = (
                f"  Constructions: {model.vn_constructed_cells} "
                f"(sensitized {model.vn_sensitized_cells}, killed {model.vn_killed_cells})"
            )
        else:
            extra = (
                f"  Gliders: {model.gliders_injected_count} injected, "
                f"{model.gliders_in_flight} in flight, "
                f"{model.gliders_consumed_count} consumed; "
                f"components {model.component_counts}"
            )
        print(f"  Result: {status} in {elapsed_ms:.2f} ms")
        print(f"  Pattern Match: {val.match_ratio * 100:.1f}%")
        print(f"  Population: {val.population} (Expected: {val.expected_population})")
        print(f"  Cost: {preset.validation_step} gens x {h}x{w} = "
              f"{cell_updates} cell-updates, {mus_per_s:.2f} Mcell-updates/s")
        print(extra)
        print(f"  Proof: {val.proof_message}")

        results.append({
            "id": preset_id,
            "name": preset.name,
            "category": preset.category,
            "step": model.current_step,
            "match": f"{val.match_ratio * 100:.1f}%",
            "population": val.population,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 2),
            "cell_updates": cell_updates,
        })

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY REPORT")
    print("=" * 70)
    print(f"{'ID':<26} | {'Status':<7} | {'Match':<7} | {'Steps':<6} | {'Time (ms)':<9}")
    print("-" * 70)
    for r in results:
        print(f"{r['id']:<26} | {r['status']:<7} | {r['match']:<7} | {r['step']:<6} | {r['elapsed_ms']:<9}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
