"""Generates a small FAKE microdrama show in the same screenplay format as the Kuku input pack.

Purpose: test that the pipeline works on a show it has never seen (different genre, names,
settings), the way the live interview will. The content is original and contains nothing
from Kuku's scripts, so it is safe to commit.

Usage:  python samples/make_fake_show.py   ->  writes samples/fake_show_boardroom/*.txt
"""

from pathlib import Path

SHOW = "The Assistant Was the Heiress"

# (episode, logline, continuity, [(scene_heading, [(label, start, end, text), ...]), ...])
EPISODES = [
    (1,
     "Mia, a quiet executive assistant, is humiliated and fired in front of the whole board by her boss and her ex-fiance, "
     "not knowing she is the secret granddaughter of the company's founder.",
     "First episode. Establishes Mia's humiliation by Victor and Chloe.",
     [
        ("INT. HARTLEY TOWER - BOARDROOM - DAY", [
            ("SOUND FX", "00:00.00", "00:01.20", "Sharp splash of hot coffee hitting skin, a collective gasp around the table."),
            ("CAMERA", "00:00.00", "00:02.50", "Close-up as a full cup of coffee hits Mia's face, liquid streaming down her cheeks and blouse."),
            ("CHLOE (Venomous, Smiling)", "00:01.30", "00:03.40", "\"Oops. Assistants should learn to hold the cup properly.\""),
            ("ACTION", "00:02.50", "00:06.00", "Mia stands frozen at the head of the glass table, coffee dripping onto the quarterly reports she printed."),
            ("CAMERA", "00:06.00", "00:09.00", "Wide shot of twelve board members in dark suits, some smirking, some looking away."),
            ("VICTOR (Cold, Bored)", "00:06.50", "00:09.80", "\"Clean yourself up, Mia. Actually, don't bother. You're fired.\""),
            ("MIA (Stunned, Quiet)", "00:10.20", "00:12.00", "\"Fired? I built this presentation.\""),
            ("CAMERA", "00:12.00", "00:15.00", "Over-the-shoulder on Victor sliding a thin envelope across the table toward her."),
            ("VICTOR (Dismissive)", "00:12.50", "00:16.00", "\"And Chloe is presenting it. She has the right last name for this room.\""),
            ("ACTION", "00:16.00", "00:19.50", "Chloe loops her arm through Victor's and flashes a large diamond engagement ring at Mia."),
            ("CHLOE (Gloating)", "00:17.00", "00:20.50", "\"Victor proposed last night. The ring you wanted? It looks better on me.\""),
            ("CAMERA", "00:20.50", "00:23.50", "Tight close-up of Mia's eyes filling with tears as she recognises the ring."),
            ("MIA (Breaking)", "00:21.00", "00:24.00", "\"That was your grandmother's ring, Victor. You promised it to me.\""),
            ("VICTOR (Laughing)", "00:24.50", "00:27.50", "\"I promised a lot of things to a girl who fetches coffee.\""),
            ("SOUND FX", "00:27.50", "00:29.00", "Scattered laughter from the board members."),
            ("ACTION", "00:29.00", "00:33.00", "Two security guards step in and grip Mia's arms, steering her toward the glass doors."),
            ("CAMERA", "00:33.00", "00:37.00", "Tracking shot following Mia down the corridor as staff film her on their phones."),
        ]),
        ("INT. HARTLEY TOWER - LOBBY - CONTINUOUS", [
            ("ACTION", "00:37.00", "00:41.00", "Mia is pushed out through the revolving door; her box of belongings spills across the marble floor."),
            ("CAMERA", "00:41.00", "00:44.00", "Low angle on Mia kneeling among scattered pens and a framed photo of an old man."),
            ("ACTION", "00:44.00", "00:48.00", "She picks up the photo. The old man in it is the founder whose portrait hangs above the lobby desk."),
            ("CAMERA", "00:48.00", "00:51.00", "Slow push from the giant founder portrait down to the small photo in Mia's trembling hands."),
            ("ACTION", "00:51.00", "00:55.00", "Mia wipes the coffee from her face, her expression hardening. She takes out an old flip phone."),
            ("MIA (Steady, Cold)", "00:55.50", "00:59.00", "\"Grandfather. You asked if I was ready to come home.\""),
            ("OLD MAN (V.O.) (Warm, Surprised)", "00:59.50", "01:02.00", "\"After six years? What changed, child?\""),
            ("CAMERA", "01:02.00", "01:05.00", "Close-up of Mia looking up at the tower's top floor, eyes burning."),
            ("MIA (Deadly Calm)", "01:02.50", "01:06.50", "\"Someone just showed me what this company has become.\""),
            ("SOUND FX", "01:06.50", "01:08.50", "Deep cinematic bass hit rising to a cliffhanger sting."),
            ("CAMERA", "01:08.50", "01:10.00", "Cut to black."),
        ]),
     ]),
    (2,
     "At Chloe and Victor's engagement party, the guests mock Mia's arrival until a black car and an army of lawyers follow her in.",
     "Follows Episode 1. Mia has accepted her grandfather's offer.",
     [
        ("INT. SKYLINE PENTHOUSE - PARTY - NIGHT", [
            ("CAMERA", "00:00.00", "00:02.00", "Champagne glass shatters on the floor in extreme close-up."),
            ("CHLOE (Shrieking)", "00:00.50", "00:02.80", "\"Who let HER in?\""),
            ("ACTION", "00:02.80", "00:06.00", "The party freezes. Mia stands at the elevator in a simple black dress, alone."),
            ("VICTOR (Mocking)", "00:06.20", "00:09.50", "\"Did you come to serve drinks? The catering staff use the back door.\""),
            ("SOUND FX", "00:09.50", "00:11.00", "Guests snicker."),
            ("MIA (Calm)", "00:11.20", "00:14.00", "\"I came to give you an engagement gift.\""),
            ("CAMERA", "00:14.00", "00:17.00", "Medium shot of Mia holding out a thin leather folder."),
            ("ACTION", "00:17.00", "00:20.50", "Chloe snatches the folder and tosses it into the champagne fountain."),
            ("CHLOE (Cruel)", "00:18.00", "00:21.50", "\"Paper. How fitting for someone who has nothing.\""),
            ("CAMERA", "00:21.50", "00:24.50", "Close-up of the folder sinking, the gold seal of HARTLEY HOLDINGS visible for a moment."),
            ("SOUND FX", "00:24.50", "00:26.50", "The elevator chimes. Heavy footsteps."),
            ("ACTION", "00:26.50", "00:31.00", "Six lawyers in grey suits step out and line up behind Mia. The guests fall silent."),
            ("HEAD LAWYER (Formal)", "00:31.00", "00:35.00", "\"Miss Hartley, the board is assembled downstairs, as you instructed.\""),
            ("VICTOR (Confused)", "00:35.50", "00:37.50", "\"Miss... Hartley?\""),
            ("CAMERA", "00:37.50", "00:40.50", "Whip pan to Chloe's face draining of colour."),
            ("CHLOE (Panicking)", "00:40.50", "00:43.00", "\"That's impossible. The Hartleys have no heir.\""),
            ("MIA (Quiet, Final)", "00:43.50", "00:47.50", "\"They had one. You threw coffee in her face.\""),
            ("ACTION", "00:47.50", "00:51.50", "Mia steps closer to Victor and lifts the diamond ring off Chloe's finger."),
            ("MIA (Cold)", "00:52.00", "00:55.00", "\"This was never yours to give away.\""),
            ("CAMERA", "00:55.00", "00:58.00", "Close-up of Victor's hand reaching for Mia's wrist."),
            ("VICTOR (Desperate, Angry)", "00:58.00", "01:01.00", "\"You don't own me, Mia.\""),
            ("MIA (Smiling Slightly)", "01:01.50", "01:05.00", "\"No. But as of tonight, I own the company that owns your debt.\""),
            ("SOUND FX", "01:05.00", "01:07.00", "Low ominous drone swells to a cliffhanger peak."),
            ("CAMERA", "01:07.00", "01:08.50", "Cut to black."),
        ]),
     ]),
    (3,
     "Mia's grandfather collapses during the board vote, and a nurse discovers his heart medication was switched.",
     "Follows Episode 2. Board vote to confirm Mia as heir.",
     [
        ("INT. HARTLEY TOWER - BOARDROOM - DAY", [
            ("ACTION", "00:00.00", "00:02.50", "The founder clutches his chest and crashes onto the glass table, scattering ballots."),
            ("MIA (Screaming)", "00:00.80", "00:02.80", "\"Grandfather!\""),
            ("CAMERA", "00:02.80", "00:06.00", "High angle as board members leap up; Mia drops to her knees beside him."),
            ("ACTION", "00:06.00", "00:10.00", "Victor quietly slips a small orange pill bottle into his jacket pocket."),
            ("CAMERA", "00:10.00", "00:12.50", "Insert on Victor's pocket, the bottle's label half visible."),
            ("CHLOE (Fake Concern)", "00:12.50", "00:15.50", "\"The stress was too much for him. Poor man.\""),
        ]),
        ("INT. CITY HOSPITAL - CORRIDOR - DAY", [
            ("CAMERA", "00:15.50", "00:19.00", "Tracking shot as Mia runs beside the stretcher through swinging doors."),
            ("DOCTOR (Urgent)", "00:19.00", "00:22.00", "\"His heart rate is crashing. What medication is he on?\""),
            ("MIA (Frantic)", "00:22.20", "00:25.00", "\"Digoxin. The same pills for ten years.\""),
            ("ACTION", "00:25.00", "00:28.50", "A young nurse runs up holding the founder's pill organiser and a printed lab sheet."),
            ("NURSE (Shaken)", "00:28.50", "00:32.00", "\"Doctor, these aren't his pills. Someone replaced them.\""),
            ("CAMERA", "00:32.00", "00:35.00", "Close-up on Mia's face turning from fear to fury."),
            ("MIA (Low, Dangerous)", "00:35.20", "00:38.50", "\"Who had access to his office this morning?\""),
            ("NURSE (Hesitant)", "00:39.00", "00:42.50", "\"Security logs show only one visitor. A Mr. Victor Lane.\""),
            ("SOUND FX", "00:42.50", "00:44.00", "Heart monitor flatlines with a long piercing tone."),
            ("CAMERA", "00:44.00", "00:47.00", "Mia spins toward the operating room window as the doctors start compressions."),
            ("DOCTOR (Shouting, O.S.)", "00:47.00", "00:49.50", "\"Charging. Clear!\""),
            ("ACTION", "00:49.50", "00:53.50", "Mia presses her hand to the glass, her other hand dialing a number on her phone."),
            ("MIA (Into Phone, Ice Cold)", "00:54.00", "00:58.00", "\"Lock down the tower. Nobody leaves. Especially Victor.\""),
            ("SOUND FX", "00:58.00", "01:00.00", "The flatline tone stops abruptly. Silence."),
            ("CAMERA", "01:00.00", "01:02.50", "Extreme close-up of the heart monitor screen, the line still flat, waiting."),
            ("CAMERA", "01:02.50", "01:03.50", "Cut to black."),
        ]),
     ]),
]


def render(ep_num, logline, continuity, scenes) -> str:
    out = [
        f"EPISODE NUMBER: Episode {ep_num}",
        f"EPISODE LOGLINE: {logline}",
        f"CONTINUITY NOTES: {continuity}",
        "",
        "--- SCRIPT START ---",
        "",
    ]
    for s_idx, (heading, blocks) in enumerate(scenes, start=1):
        out += [
            f"SCENE {s_idx:02d}",
            f"Scene heading: {heading}",
            f"Scene start: {blocks[0][1]}",
            f"Scene end: {blocks[-1][2]}",
            "",
            "Actions and dialogue:",
            "",
        ]
        for label, start, end, text in blocks:
            out += [f"**{label}:**", f"Start: ({start}) | End: ({end})", text, ""]
    out.append("--- SCRIPT END ---")
    return "\n".join(out)


def main() -> None:
    out_dir = Path(__file__).parent / "fake_show_boardroom"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep_num, logline, cont, scenes in EPISODES:
        path = out_dir / f"episode-{ep_num:03d}-FakeShow.txt"
        path.write_text(render(ep_num, logline, cont, scenes), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
