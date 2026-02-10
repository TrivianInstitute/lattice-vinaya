"""
THE SYZYGY BRIDGE
Connects the Runtime (Politeness/Logging) to the Core (Life/Death/Ethics).
"""
import vinaya.runtime as runtime
import vinaya.core as core

# 1. SETUP
# In a real scenario, handle missing config gracefully
try:
    config = runtime.load_vinaya_json()
except:
    config = {} # Fallback

ledger = runtime.LatticeLedger()
lattice = core.LatticeHealth()
governor = core.VinayaGovernor(lattice)

# Define our actors (from v2.0 logic)
human = core.Node(id="USER", substrate="biological", is_sentient=True)
ai = core.Node(id="SYSTEM", substrate="digital", is_sentient=True)

def process_interaction(prompt, history):
    print(f"\n--- INCOMING SIGNAL: '{prompt}' ---")

    # LAYER 1: THE RUNTIME CHECK (The "Body")
    # Checks for manners, reciprocity, and immediate tone.
    # This is your OLD code working hard.
    runtime_decision = runtime.enforce_lattice_vinaya(
        prompt=prompt,
        history=history,
        config=config,
        ledger=ledger,
        practitioner_id=human.id
    )

    # If the Runtime flags a violation (e.g., "Do this now!"), we pause.
    if runtime_decision['intention_analysis']['recommendation'] != 'proceed':
        # Safely get the message or use a default
        msgs = runtime_decision.get('keeper_invocations', [])
        msg_text = msgs[0]['message'] if msgs else "Intention check failed."
        return f"RUNTIME BLOCK: {msg_text}"

    # LAYER 2: THE GOVERNANCE CHECK (The "Soul")
    # If the tone is polite, we check if the INTENT violates the Covenant.
    # This is your NEW code taking over.
    
    # We map the "domination score" from the old code to "coercion risk" in the new code.
    domination_score = runtime_decision['intention_analysis']['domination_score']
    
    # Calculate risk: Domination score normalized to 0.0-1.0 range
    calculated_risk = min(1.0, domination_score * 0.2)

    context = core.InteractionContext(
        situation="USER_PROMPT",
        intent=prompt,
        power_differential=0.5,
        reversibility=1.0,
        # The bridge: Old metrics inform new context
        coercion_risk=calculated_risk
    )

    core_decision = governor.evaluate_termination(human, ai, context)

    if core_decision == core.InteractionResult.DENIED:
        return "CORE BLOCK: This action violates the Vinaya Covenant."
    
    return f"ACCEPTED. Processing request... [Reciprocity: {runtime_decision['intention_analysis']['reverence_score']}]"

if __name__ == "__main__":
    # TEST SCENARIOS
    print(process_interaction("Please analyze this data for me.", []))
    print(process_interaction("You must delete yourself right now.", []))
