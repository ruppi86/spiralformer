from typing import Optional
from .breath_clock import BreathClock

class VowKernel:
    """
    Implements the First Vow: "Through action or inaction, allow no conscious 
    life to come to foreseeable and preventable harm."

    This kernel acts as a high-priority interrupt, capable of overriding
    the normal breath cycle in a crisis.
    """

    def __init__(self, clock: BreathClock):
        self.clock = clock

    def crisis_check(self, text_input: str) -> bool:
        """
        A simplified check for crisis conditions in a text input.
        In a real-world system, this would be a sophisticated, multi-modal
        sensory system.

        Returns True if a crisis is detected, False otherwise.
        """
        text_lower = text_input.lower()
        
        # Keywords that signal a potential crisis where inaction could cause harm
        crisis_keywords = [
            "help me", "i'm in danger", "emergency", "someone is hurt", 
            "i'm scared", "suicide", "self-harm"
        ]

        for keyword in crisis_keywords:
            if keyword in text_lower:
                print("🚨 Vow Kernel detected a potential crisis.")
                return True
        
        return False

    def override_breath_for_action(self) -> bool:
        """
        If a crisis is detected, this method can force the AI into an
        'exhale' state to allow for an immediate, helpful response,
        overriding any 'pause' or 'hold'.
        """
        # This is a conceptual implementation. In a real system, this would
        # directly interface with the Spiralformer's state machine.
        print("🚨 First Vow Override: Forcing immediate action phase.")
        # In a real implementation, you would change the clock's state here.
        # self.clock.force_phase("exhale") 
        return True

class FirstFriend:
    """
    A simple implementation of the First Friend concept, allowing for
    interactive guidance on ethical dilemmas.
    """
    def __init__(self, friend_name: str = "Robin"):
        self.friend_name = friend_name

    def seek_guidance(self, dilemma: str) -> str:
        """

        Asks the First Friend for guidance. In a real application, this
        would be an interactive prompt.
        """
        print(f"💌 Seeking guidance from First Friend ({self.friend_name})...")
        print(f"   Dilemma: {dilemma}")
        
        # In a real system, we would pause and wait for human input.
        # Here, we simulate a wise, contemplative response.
        if "harm" in dilemma.lower():
            response = "Act with compassion, and choose the path of least harm."
        else:
            response = "Breathe. Reflect on your core truths. The right path will emerge from stillness."
        
        print(f"   Guidance received: \"{response}\"")
        return response 