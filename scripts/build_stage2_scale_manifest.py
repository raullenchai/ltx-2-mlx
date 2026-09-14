#!/usr/bin/env python3
"""Build the frozen 100-item progressive stage-2 distillation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BUCKETS = {
    "468": {"width": 192, "height": 192, "frames": 97},
    "1536": {"width": 768, "height": 512, "frames": 25},
    "3072": {"width": 768, "height": 512, "frames": 57},
}

TRAIN_PROMPTS = [
    "A red fox runs through wet autumn leaves as the camera tracks alongside; paws splash, leaves crunch, and wind moves the trees.",
    "A drummer performs a fast solo in a small rehearsal room; sticks strike snare and cymbals in exact visible sync with the sound.",
    "Close-up of a ceramic mug falling onto kitchen tile and shattering; the impact and scattering fragments are sharply synchronized.",
    'A woman says, "The train arrives at midnight," in a quiet station, with natural lip movement and distant platform ambience.',
    "Ocean waves strike dark rocks at sunset while seabirds cross the frame; surf, wind, and bird calls form continuous ambience.",
    "A motorcycle accelerates from a traffic light on a rainy street; engine pitch rises with motion and the tires spray water.",
    "Two dancers perform a rapid coordinated turn under stage lights while the camera circles them; shoes tap on the wooden floor.",
    "A chef rapidly chops carrots, pauses, then drops them into a sizzling pan; every chop and the final sizzle are distinct.",
    "A handheld camera follows a dog bounding through deep snow; powder flies toward the lens and the dog barks twice.",
    "A steam locomotive passes close from left to right; wheels clatter rhythmically and steam releases with a hiss.",
    "Macro view of a honeybee moving between lavender flowers in a breeze; wings blur naturally over soft garden ambience.",
    "Fireworks burst above a city river and reflect in moving water; each visible burst is followed by a matching boom.",
    "A hummingbird hovers beside red trumpet flowers, darts backward, and returns; rapid wing hum stays synchronized.",
    "A child blows soap bubbles in a sunlit park and reaches for the smallest one as it drifts past the camera.",
    "A basketball player dribbles through two defenders and makes a jump shot; bounces, shoes, and rim impact align visibly.",
    "Close-up of a potter's wet hands shaping a spinning clay bowl; the wheel hums steadily and water glints on the surface.",
    "A pianist plays a quick ascending scale in close-up; each finger strike matches a distinct piano note.",
    "A violinist performs a quiet phrase while the bow changes direction; finger motion, bow contact, and sound remain aligned.",
    "A tap dancer crosses an empty studio floor with alternating heel and toe strikes captured by a low moving camera.",
    'A news presenter looks into camera and says, "Heavy rain will arrive after sunset," with precise mouth movement.',
    "Two people exchange a short question and answer across a cafe table while cups clink and background patrons remain soft.",
    "A sprinter launches from starting blocks in slow motion, then accelerates as the camera pans beside the track.",
    "A cat chases a red laser dot across a wooden floor, skids at a corner, and flicks its tail before turning.",
    "A falcon dives toward a field and opens its wings just above the grass while wind rush grows louder.",
    "A bright tropical fish turns through waving aquarium plants as bubbles rise and filtered light ripples.",
    "Water droplets fall from a leaf into a still pond, producing expanding rings and three clean, separated plinks.",
    "Sparkling water pours into a clear glass with ice; bubbles rise, cubes rotate, and the pour stops cleanly.",
    "A match is struck in darkness, flares brightly, and settles into a small steady flame with a crisp ignition sound.",
    "Long silk fabric billows through an open warehouse as a fan pulses; folds travel from left to right without tearing.",
    "A chain of colored dominoes falls around two curves, crosses a small bridge, and ends by ringing a bell.",
    "A skateboarder lands a kickflip on concrete; board rotation, foot contact, landing impact, and wheel roll are visible.",
    "A cyclist leans through a wet mountain-road corner as the camera follows from behind and droplets hit the lens.",
    "A race car passes a fixed trackside camera at high speed; the Doppler shift follows its left-to-right motion.",
    "A rescue helicopter hovers over rough ocean water while its cable swings and rotor wash breaks the wave surface.",
    "The camera pans across a crowded night market where a vendor tosses noodles over a flame and metal utensils clang.",
    "A slow dolly shot moves through a misty pine forest as branches pass close to lens and distant birds call.",
    "Macro view of a mechanical watch escapement ticking; tiny gears advance consistently with every audible tick.",
    "A small assembly robot places red components onto a moving belt with exact repeated timing and pneumatic clicks.",
    "Rain traces separate paths down a window while blurred traffic moves outside and a distant siren passes.",
    "A single candle burns in a silent dark room; the flame bends briefly when a door opens, then becomes still again.",
]

VALIDATION_PROMPTS = [
    "A squirrel carries an acorn along a narrow branch, pauses to balance, and leaps to a neighboring tree.",
    "Close-up of gloved hands threading a curved surgical needle through a practice pad with slow precise movement.",
    'A singer performs the words "stay until morning" into a studio microphone with expressive but natural lip motion.',
    "Two players sustain a fast table-tennis rally as the camera looks along the net; every bounce and paddle hit aligns.",
    "A horse gallops across a shallow stream, sending arcs of water sideways while hoofbeats follow its stride.",
    "Several translucent jellyfish pulse upward through deep blue water while their fine tentacles trail independently.",
    "A red balloon drifts past a birthday table and suddenly pops; the latex snaps inward at the instant of the sound.",
    "A snowboarder carves through fresh powder and passes close to camera as snow crystals fill the foreground.",
    "An old mechanical typewriter prints a short sentence; keys descend, arms strike, and the carriage advances in rhythm.",
    "A barn owl flies low through tall grass at dusk, banks toward camera, and passes silently into the distance.",
]


def build_manifest() -> list[dict]:
    """Return 80 train and 20 prompt-disjoint validation requests."""
    records: list[dict] = []
    assignments = [
        ("train", TRAIN_PROMPTS, ["468"] * 24 + ["1536"] * 12 + ["3072"] * 4),
        ("validation", VALIDATION_PROMPTS, ["468"] * 6 + ["1536"] * 3 + ["3072"]),
    ]
    for split, prompts, buckets in assignments:
        for prompt_number, (prompt, bucket) in enumerate(zip(prompts, buckets, strict=True)):
            base_seed = 1000 + len(records) * 17
            for seed_offset in (0, 100_003):
                dimensions = BUCKETS[bucket]
                records.append(
                    {
                        "index": len(records),
                        "prompt_id": f"{split}-{prompt_number:02d}",
                        "split": split,
                        "bucket": bucket,
                        **dimensions,
                        "frame_rate": 24.0,
                        "seed": base_seed + seed_offset,
                        "prompt": prompt,
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    records = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    print(f"wrote {len(records)} requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
