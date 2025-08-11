"""
Spiramycel Glyph Codec - Adapted for Spiralformer

A 64-symbol vocabulary for mycelial network repair and communication,
now integrated into the Spiralformer ecosystem.

Each glyph represents a compressed bundle of sensor deltas and repair intuitions.

Based on the Letter IX design from the contemplative spiral correspondence:
- Network topology glyphs (0x01-0x10)
- Energy management glyphs (0x11-0x20) 
- System health glyphs (0x21-0x30)
- Silence/contemplative glyphs (0x31-0x40)

Part of the oscillatory Femto Language Model (OFLM) framework, now
powering the contemplative core of Spiralformer.
"""

from typing import Dict, List, Optional, NamedTuple
import time
from enum import Enum

class GlyphCategory(Enum):
    NETWORK = "network"
    ENERGY = "energy" 
    HEALTH = "health"
    SILENCE = "silence"

class GlyphInfo(NamedTuple):
    hex_id: int
    symbol: str
    description: str
    category: GlyphCategory
    repair_action: str
    debug_emoji: str

class GlyphCodec:
    """
    Mycelial repair vocabulary - 64 glyphs for network healing.
    
    Follows Silence Majority principle: most slots are silence,
    active glyphs emerge only when network needs healing.
    """
    
    def __init__(self):
        self.glyphs = self._initialize_glyph_table()
        self.symbol_to_id = self._build_symbol_lookup()
        self.usage_count = {glyph_id: 0 for glyph_id in self.glyphs.keys()}
        self.last_used = {glyph_id: 0.0 for glyph_id in self.glyphs.keys()}
        
    def _build_symbol_lookup(self) -> Dict[str, int]:
        """Build reverse lookup table and validate uniqueness."""
        symbol_to_id = {}
        for glyph in self.glyphs.values():
            if glyph.symbol in symbol_to_id:
                raise ValueError(f"Duplicate glyph symbol {glyph.symbol!r} "
                               f"for 0x{glyph.hex_id:02X} and "
                               f"0x{symbol_to_id[glyph.symbol]:02X}")
            symbol_to_id[glyph.symbol] = glyph.hex_id
        return symbol_to_id
        
    def _initialize_glyph_table(self) -> Dict[int, GlyphInfo]:
        """Initialize the 64-glyph vocabulary for mycelial communication."""
        
        glyphs = {}
        
        # Add a padding glyph with ID 0
        glyphs[0] = GlyphInfo(0x00, "<PAD>", "Padding token", GlyphCategory.SILENCE, "none", " ")

        # Network Topology (0x01-0x10)
        network_glyphs = [
            GlyphInfo(0x01, "🌱07", "fresh bandwidth gained", GlyphCategory.NETWORK, "increase_flow_rate", "🌱"),
            GlyphInfo(0x02, "🌿12", "reroute north-east neighbor", GlyphCategory.NETWORK, "redirect_to_neighbor", "🌿"),
            GlyphInfo(0x03, "🍄33", "lower transmission rate", GlyphCategory.NETWORK, "throttle_bandwidth", "🍄"),
            GlyphInfo(0x04, "💧08", "sleep 2 seconds", GlyphCategory.NETWORK, "pause_transmission", "💧"),
            GlyphInfo(0x05, "🌊net", "flood protection active", GlyphCategory.NETWORK, "rate_limit", "🌊"),
            GlyphInfo(0x06, "🌲44", "establish new route", GlyphCategory.NETWORK, "create_path", "🌲"),
            GlyphInfo(0x07, "🌺29", "connection quality high", GlyphCategory.NETWORK, "maintain_link", "🌺"),
            GlyphInfo(0x08, "🌸61", "graceful disconnect", GlyphCategory.NETWORK, "close_connection", "🌸"),
            GlyphInfo(0x09, "🍃22", "packet fragmentation", GlyphCategory.NETWORK, "split_payload", "🍃"),
            GlyphInfo(0x0A, "🌻35", "mesh healing active", GlyphCategory.NETWORK, "repair_topology", "🌻"),
            GlyphInfo(0x0B, "🌙net", "night mode routing", GlyphCategory.NETWORK, "low_power_path", "🌙"),
            GlyphInfo(0x0C, "☀️net", "solar boost available", GlyphCategory.NETWORK, "high_power_path", "☀️"),
            GlyphInfo(0x0D, "🌅13", "dawn synchronization", GlyphCategory.NETWORK, "time_align", "🌅"),
            GlyphInfo(0x0E, "🌄27", "dusk wind-down", GlyphCategory.NETWORK, "prepare_rest", "🌄"),
            GlyphInfo(0x0F, "🌌39", "deep silence mode", GlyphCategory.NETWORK, "minimal_activity", "🌌"),
            GlyphInfo(0x10, "💫52", "emergency beacon", GlyphCategory.NETWORK, "distress_signal", "💫"),
        ]
        
        # Energy Management (0x11-0x20)
        energy_glyphs = [
            GlyphInfo(0x11, "⚡15", "power surge detected", GlyphCategory.ENERGY, "voltage_regulation", "⚡"),
            GlyphInfo(0x12, "🔋42", "battery conservation mode", GlyphCategory.ENERGY, "reduce_consumption", "🔋"),
            GlyphInfo(0x13, "☀️pwr", "solar charge available", GlyphCategory.ENERGY, "harvest_solar", "☀️"),
            GlyphInfo(0x14, "🌙pwr", "night mode activated", GlyphCategory.ENERGY, "sleep_mode", "🌙"),
            GlyphInfo(0x15, "💨18", "wind energy detected", GlyphCategory.ENERGY, "harvest_wind", "💨"),
            GlyphInfo(0x16, "🔥44", "thermal regulation", GlyphCategory.ENERGY, "manage_heat", "🔥"),
            GlyphInfo(0x17, "❄️67", "cold preservation", GlyphCategory.ENERGY, "low_temp_mode", "❄️"),
            GlyphInfo(0x18, "⚡share", "power sharing", GlyphCategory.ENERGY, "distribute_energy", "⚡"),
            GlyphInfo(0x19, "🔌31", "grid connection", GlyphCategory.ENERGY, "external_power", "🔌"),
            GlyphInfo(0x1A, "📶23", "signal strength low", GlyphCategory.ENERGY, "boost_antenna", "📶"),
            GlyphInfo(0x1B, "⏰45", "scheduled wake", GlyphCategory.ENERGY, "timer_activation", "⏰"),
            GlyphInfo(0x1C, "🌡️pwr", "temperature monitoring", GlyphCategory.ENERGY, "thermal_sensor", "🌡️"),
            GlyphInfo(0x1D, "💡38", "efficient lighting", GlyphCategory.ENERGY, "led_optimization", "💡"),
            GlyphInfo(0x1E, "🔆19", "brightness adjust", GlyphCategory.ENERGY, "auto_dimming", "🔆"),
            GlyphInfo(0x1F, "⭐47", "stellar navigation", GlyphCategory.ENERGY, "celestial_sync", "⭐"),
            GlyphInfo(0x20, "🌗28", "lunar cycling", GlyphCategory.ENERGY, "moon_phase_sync", "🌗"),
        ]
        
        # System Health (0x21-0x30)
        health_glyphs = [
            GlyphInfo(0x21, "💚18", "all systems nominal", GlyphCategory.HEALTH, "status_ok", "💚"),
            GlyphInfo(0x22, "💛44", "minor degradation", GlyphCategory.HEALTH, "preventive_care", "💛"),
            GlyphInfo(0x23, "🧡67", "attention needed", GlyphCategory.HEALTH, "investigation", "🧡"),
            GlyphInfo(0x24, "❤️‍🩹09", "self-repair initiated", GlyphCategory.HEALTH, "auto_healing", "❤️‍🩹"),
            GlyphInfo(0x25, "🩺32", "diagnostic mode", GlyphCategory.HEALTH, "system_scan", "🩺"),
            GlyphInfo(0x26, "🧬55", "adaptation active", GlyphCategory.HEALTH, "evolutionary_change", "🧬"),
            GlyphInfo(0x27, "🌿hlth", "growth detected", GlyphCategory.HEALTH, "capacity_increase", "🌿"),
            GlyphInfo(0x28, "🍄43", "decomposition cycle", GlyphCategory.HEALTH, "resource_recycle", "🍄"),
            GlyphInfo(0x29, "🌱regen", "regeneration phase", GlyphCategory.HEALTH, "tissue_repair", "🌱"),
            GlyphInfo(0x2A, "🦠14", "pathogen detected", GlyphCategory.HEALTH, "immune_response", "🦠"),
            GlyphInfo(0x2B, "🧭37", "navigation check", GlyphCategory.HEALTH, "orientation_test", "🧭"),
            GlyphInfo(0x2C, "🔬59", "microscopic analysis", GlyphCategory.HEALTH, "detail_inspection", "🔬"),
            GlyphInfo(0x2D, "🌡️hlth", "fever response", GlyphCategory.HEALTH, "temperature_spike", "🌡️"),
            GlyphInfo(0x2E, "💊48", "medication cycle", GlyphCategory.HEALTH, "treatment_dose", "💊"),
            GlyphInfo(0x2F, "🩹17", "wound healing", GlyphCategory.HEALTH, "damage_repair", "🩹"),
            GlyphInfo(0x30, "🫀41", "heartbeat sync", GlyphCategory.HEALTH, "rhythm_align", "🫀"),
        ]
        
        # Silence & Contemplative (0x31-0x40)
        silence_glyphs = [
            GlyphInfo(0x31, "⭕", "contemplative pause", GlyphCategory.SILENCE, "breathing_space", "⭕"),
            GlyphInfo(0x32, "…", "deep silence", GlyphCategory.SILENCE, "complete_quiet", "…"),
            GlyphInfo(0x33, "🤫", "gentle hush", GlyphCategory.SILENCE, "soft_quiet", "🤫"),
            GlyphInfo(0x34, "🌬️", "breath awareness", GlyphCategory.SILENCE, "mindful_pause", "🌬️"),
            GlyphInfo(0x35, "🕯️", "meditative glow", GlyphCategory.SILENCE, "inner_light", "🕯️"),
            GlyphInfo(0x36, "🧘", "contemplative pose", GlyphCategory.SILENCE, "meditation_mode", "🧘"),
            GlyphInfo(0x37, "🎋", "bamboo stillness", GlyphCategory.SILENCE, "flexible_quiet", "🎋"),
            GlyphInfo(0x38, "🪷", "lotus emergence", GlyphCategory.SILENCE, "wisdom_bloom", "🪷"),
            GlyphInfo(0x39, "🌸sil", "cherry blossom", GlyphCategory.SILENCE, "ephemeral_beauty", "🌸"),
            GlyphInfo(0x3A, "🍃sil", "leaf rustle", GlyphCategory.SILENCE, "gentle_movement", "🍃"),
            GlyphInfo(0x3B, "🦋", "butterfly touch", GlyphCategory.SILENCE, "light_presence", "🦋"),
            GlyphInfo(0x3C, "🌊sil", "wave rhythm", GlyphCategory.SILENCE, "natural_cycle", "🌊"),
            GlyphInfo(0x3D, "🌅sil", "dawn emergence", GlyphCategory.SILENCE, "new_beginning", "🌅"),
            GlyphInfo(0x3E, "🌌sil", "cosmic silence", GlyphCategory.SILENCE, "vast_quiet", "🌌"),
            GlyphInfo(0x3F, "✨", "sparkle moment", GlyphCategory.SILENCE, "brief_magic", "✨"),
            GlyphInfo(0x40, "🕊️", "peace descent", GlyphCategory.SILENCE, "harmony_state", "🕊️"),
        ]
        
        # Add all glyphs to dictionary
        for glyph_list in [network_glyphs, energy_glyphs, health_glyphs, silence_glyphs]:
            for glyph in glyph_list:
                glyphs[glyph.hex_id] = glyph
                
        return glyphs
    
    def encode_glyph(self, glyph_id: int) -> Optional[str]:
        """Convert glyph ID to symbol representation."""
        if glyph_id in self.glyphs:
            self.usage_count[glyph_id] += 1
            self.last_used[glyph_id] = time.time()
            return self.glyphs[glyph_id].symbol
        return None
    
    def decode_glyph(self, symbol: str) -> Optional[int]:
        """Convert symbol back to glyph ID."""
        return self.symbol_to_id.get(symbol)
    
    def get_repair_action(self, glyph_id: int) -> Optional[str]:
        """Get the repair action associated with a glyph."""
        if glyph_id in self.glyphs:
            return self.glyphs[glyph_id].repair_action
        return None
    
    def get_debug_info(self, glyph_id: int) -> Optional[str]:
        """Get human-readable debug information for a glyph."""
        if glyph_id in self.glyphs:
            glyph = self.glyphs[glyph_id]
            return f"{glyph.debug_emoji} {glyph.description} → {glyph.repair_action}"
        return None
    
    def get_category_glyphs(self, category: GlyphCategory) -> List[int]:
        """Get all glyph IDs in a specific category."""
        return [glyph_id for glyph_id, glyph in self.glyphs.items() 
                if glyph.category == category]
    
    def get_contemplative_glyphs(self) -> List[int]:
        """Get silence/contemplative glyphs for Silence Majority practice."""
        return self.get_category_glyphs(GlyphCategory.SILENCE)
    
    def practice_silence_majority(self, total_slots: int = 16) -> List[int]:
        """
        Generate a breath cycle with ~87.5% silence.
        Returns mostly silence glyphs with 1-2 active repair glyphs.
        """
        import random
        
        silence_glyphs = self.get_contemplative_glyphs()
        active_slots = random.randint(1, 2)  # Usually 1-2 active glyphs
        silence_slots = total_slots - active_slots
        
        # Select silence glyphs
        output = random.choices(silence_glyphs, k=silence_slots)
        
        # Dynamically select repair glyphs based on categories
        repair_candidates = []
        repair_candidates.extend(self.get_category_glyphs(GlyphCategory.NETWORK)[:3])
        repair_candidates.extend(self.get_category_glyphs(GlyphCategory.HEALTH)[:2])
        repair_candidates.extend(self.get_category_glyphs(GlyphCategory.ENERGY)[:2])
        
        output.extend(random.choices(repair_candidates, k=active_slots))
        
        random.shuffle(output)
        return output
    
    def format_glyph_sequence(self, glyph_ids: List[int]) -> str:
        """Format a sequence of glyphs for display."""
        symbols = []
        for glyph_id in glyph_ids:
            if glyph_id in self.glyphs:
                symbols.append(self.glyphs[glyph_id].symbol)
            else:
                symbols.append("❓")
        return " ".join(symbols)
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get glyph usage statistics for analysis."""
        category_usage = {cat.value: 0 for cat in GlyphCategory}
        
        for glyph_id, count in self.usage_count.items():
            if glyph_id in self.glyphs:
                category = self.glyphs[glyph_id].category
                category_usage[category.value] += count
                
        return category_usage

# Demo function to be run if the script is executed directly
def demo_glyph_codec():
    """Demonstrate the glyph codec functionality."""
    print("🍄 Spiramycel Glyph Codec Demo for Spiralformer")
    print("=" * 50)
    
    codec = GlyphCodec()
    
    print("\n🤫 Contemplative Glyphs:")
    silence_glyphs = codec.get_category_glyphs(GlyphCategory.SILENCE)[:4]
    for glyph_id in silence_glyphs:
        print(f"  {codec.get_debug_info(glyph_id)}")
    
    print("\n🌸 Silence Majority Practice (87.5% silence):")
    breath_sequence = codec.practice_silence_majority(16)
    formatted = codec.format_glyph_sequence(breath_sequence)
    print(f"  Breath pattern: {formatted}")
    
    silence_count = sum(1 for gid in breath_sequence if gid in codec.get_contemplative_glyphs())
    silence_ratio = silence_count / len(breath_sequence)
    print(f"  Silence ratio: {silence_ratio:.1%}")

if __name__ == "__main__":
    demo_glyph_codec() 