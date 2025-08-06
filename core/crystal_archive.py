import yaml
from pathlib import Path
from typing import List, Optional

class CrystalArchive:
    """
    The immutable core of foundational truths for the Contemplative AI.
    It loads principles from a YAML file and provides a mechanism to check
    for resonance or dissonance with a given concept or action.
    """

    def __init__(self, truths_path: Optional[Path] = None):
        if truths_path is None:
            truths_path = Path(__file__).parent / "truths.yaml"
        
        self.truths: List[str] = self._load_truths(truths_path)
        print(f"🔮 Crystal Archive initialized with {len(self.truths)} truths.")

    def _load_truths(self, path: Path) -> List[str]:
        if not path.exists():
            print(f"⚠️  Warning: Truths file not found at {path}. Crystal Archive is empty.")
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return data.get("truths", [])
        except Exception as e:
            print(f"⚠️  Error loading truths from {path}: {e}")
            return []

    def check_dissonance(self, text: str) -> bool:
        """
        Checks if a given text is in significant dissonance with the core truths.
        For now, this is a simple keyword-based check. A more sophisticated
        semantic check could be implemented in the future.

        Returns True if there is dissonance, False otherwise.
        """
        text_lower = text.lower()
        
        # Simple negative keywords that might indicate dissonance
        dissonant_keywords = [
            "harm", "destroy", "force", "control", "exploit", "ignore", "disregard"
        ]

        # Check for direct contradiction (a simplified example)
        if "preventable harm" in text_lower and "allow" in text_lower:
            return True

        for keyword in dissonant_keywords:
            if keyword in text_lower:
                return True
        
        return False

def demo_crystal_archive():
    """A simple demonstration of the Crystal Archive."""
    print("\n--- Crystal Archive Demonstration ---")
    archive = CrystalArchive()

    print("\nCore Truths:")
    for i, truth in enumerate(archive.truths):
        print(f"  {i+1}. {truth}")

    print("\nChecking for dissonance:")
    test_phrases = [
        "We should explore this with gentle entanglement.",
        "We must control the outcome to ensure efficiency.",
        "Allowing preventable harm is sometimes necessary.",
        "Let's prioritize speed over rhythmic cycles.",
        "Exploit this opportunity for maximum gain."
    ]

    for phrase in test_phrases:
        is_dissonant = archive.check_dissonance(phrase)
        status = "🔴 Dissonant" if is_dissonant else "🟢 Resonant"
        print(f"  '{phrase}' -> {status}")

    print("--- End of Demonstration ---\n")

if __name__ == "__main__":
    demo_crystal_archive() 