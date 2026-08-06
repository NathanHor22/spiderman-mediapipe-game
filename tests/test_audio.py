import unittest
from array import array
from types import SimpleNamespace

import pygame

from spidergame.audio import (
    ASSET_DIR,
    CABLE_CHANNEL,
    IMPACT_CHANNEL,
    MOTION_CHANNEL,
    MUSIC_MENU,
    OUTPUT_CHANNELS,
    SAMPLE_RATE,
    SHOT_CHANNEL,
    SOUND_FILES,
    SoundSystem,
    _build_pcm_bank,
    trim_silence,
)
from spidergame.game.swing import SwingEvent


class FakeSound:
    def __init__(self, buffer):
        self.buffer = buffer


class FakeChannel:
    def __init__(self, index):
        self.index = index
        self.plays = []
        self.busy = False
        self.volume = ()
        self.fadeouts = []
        self.stops = 0

    def play(self, sound, loops=0, fade_ms=0):
        self.plays.append((sound, loops, fade_ms))
        self.busy = True

    def get_busy(self):
        return self.busy

    def set_volume(self, *volume):
        self.volume = volume

    def fadeout(self, milliseconds):
        self.fadeouts.append(milliseconds)
        self.busy = False

    def stop(self):
        self.stops += 1
        self.busy = False


class FakeMixer:
    def __init__(self, *, ready=True, fail=False):
        self.ready = ready
        self.fail = fail
        self.init_calls = []
        self.num_channels = 2
        self.reserved = None
        self.channels = {}
        self.sounds = []

    def get_init(self):
        return (SAMPLE_RATE, -16, OUTPUT_CHANNELS) if self.ready else None

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail:
            raise pygame.error("no audio device")
        self.ready = True

    def get_num_channels(self):
        return self.num_channels

    def set_num_channels(self, count):
        self.num_channels = count

    def set_reserved(self, count):
        self.reserved = count

    def Channel(self, index):
        return self.channels.setdefault(index, FakeChannel(index))

    def Sound(self, *, buffer):
        sound = FakeSound(buffer)
        self.sounds.append(sound)
        return sound


def swinging_sim(**overrides):
    values = dict(
        attached=True,
        rope_taut=True,
        speed=62.0,
        angular_speed=1.1,
        web_tension=48.0,
        vx=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ProceduralPcmTests(unittest.TestCase):
    def test_pcm_bank_is_deterministic_stereo_and_non_silent(self):
        first = _build_pcm_bank()
        second = _build_pcm_bank()

        self.assertEqual(
            first.keys(),
            {"shot", "attach", "miss", "release", "whoosh", "impact"},
        )
        self.assertEqual(first, second)
        for pcm in first.values():
            self.assertEqual(len(pcm) % (OUTPUT_CHANNELS * 2), 0)
            self.assertGreater(len(pcm), SAMPLE_RATE // 20 * OUTPUT_CHANNELS * 2)
            samples = array("h")
            samples.frombytes(pcm)
            self.assertTrue(any(samples))
            self.assertLessEqual(max(samples), 32_767)
            self.assertGreaterEqual(min(samples), -32_767)


class SoundSystemTests(unittest.TestCase):
    def test_initialises_mixer_and_reserves_fixed_channels(self):
        mixer = FakeMixer(ready=False)
        audio = SoundSystem(mixer=mixer)

        self.assertTrue(audio.enabled)
        self.assertEqual(len(mixer.init_calls), 1)
        self.assertEqual(mixer.num_channels, 4)
        self.assertEqual(mixer.reserved, 4)
        self.assertEqual(
            set(mixer.channels),
            {SHOT_CHANNEL, CABLE_CHANNEL, MOTION_CHANNEL, IMPACT_CHANNEL},
        )

    def test_mixer_failure_degrades_to_safe_silence(self):
        mixer = FakeMixer(ready=False, fail=True)
        audio = SoundSystem(mixer=mixer)

        self.assertFalse(audio.enabled)
        self.assertIn("no audio device", audio.error)
        audio.handle((SwingEvent.SHOT, SwingEvent.ATTACH), swinging_sim())
        audio.update(swinging_sim())
        audio.stop()
        audio.close()

    def test_shot_and_attach_layer_on_separate_channels(self):
        mixer = FakeMixer()
        audio = SoundSystem(mixer=mixer)

        audio.handle((SwingEvent.SHOT, SwingEvent.ATTACH), swinging_sim())

        shot_play = mixer.channels[SHOT_CHANNEL].plays[-1]
        attach_play = mixer.channels[CABLE_CHANNEL].plays[-1]
        loop_play = mixer.channels[MOTION_CHANNEL].plays[-1]
        self.assertIs(shot_play[0], audio._sounds["shot"])
        self.assertIs(attach_play[0], audio._sounds["attach"])
        self.assertIs(loop_play[0], audio._sounds["whoosh"])
        self.assertEqual(loop_play[1:], (-1, 80))

    def test_miss_and_release_have_cues_and_release_fades_motion(self):
        mixer = FakeMixer()
        audio = SoundSystem(mixer=mixer)
        audio.handle((SwingEvent.ATTACH,), swinging_sim())

        audio.handle((SwingEvent.MISS,), swinging_sim(attached=False))
        self.assertIs(mixer.channels[CABLE_CHANNEL].plays[-1][0], audio._sounds["miss"])
        audio.handle((SwingEvent.RELEASE,), swinging_sim(attached=False))

        self.assertIs(mixer.channels[CABLE_CHANNEL].plays[-1][0], audio._sounds["release"])
        self.assertIn(100, mixer.channels[MOTION_CHANNEL].fadeouts)

    def test_loop_volume_tracks_speed_and_tautness(self):
        mixer = FakeMixer()
        audio = SoundSystem(master_volume=1.0, mixer=mixer)

        audio.update(swinging_sim(speed=20.0, angular_speed=0.05))
        low = mixer.channels[MOTION_CHANNEL].volume[0]
        audio.update(swinging_sim(speed=120.0, angular_speed=2.4))
        high = mixer.channels[MOTION_CHANNEL].volume[0]
        audio.update(swinging_sim(speed=120.0, angular_speed=2.4,
                                  rope_taut=False))
        slack = mixer.channels[MOTION_CHANNEL].volume[0]

        self.assertGreater(high, low)
        self.assertLess(slack, high)

    def test_update_stops_orphaned_loop_when_not_attached(self):
        mixer = FakeMixer()
        audio = SoundSystem(mixer=mixer)
        audio.update(swinging_sim())

        audio.update(swinging_sim(attached=False))

        self.assertIn(90, mixer.channels[MOTION_CHANNEL].fadeouts)

    def test_fall_impact_survives_the_stop_that_death_triggers(self):
        # The death transition calls stop() and then play_fall(). If the impact
        # shared a channel with the swing cues, the cleanup for dying would cut
        # off the sound of dying.
        mixer = FakeMixer()
        audio = SoundSystem(mixer=mixer)

        audio.stop()
        audio.play_fall()

        impact = mixer.channels[IMPACT_CHANNEL]
        self.assertEqual(len(impact.plays), 1)
        self.assertEqual(impact.fadeouts, [])
        self.assertTrue(impact.get_busy())

    def test_fake_mixer_never_touches_recorded_assets(self):
        # Injected mixers have no decoder, so tests must stay on the
        # synthesised bank or their buffer assertions become meaningless.
        mixer = FakeMixer()
        audio = SoundSystem(mixer=mixer)

        self.assertEqual(audio.loaded_assets, [])
        self.assertFalse(audio.music_playing)
        audio.play_menu_music()
        self.assertFalse(audio.music_playing)

    def test_shipped_audio_assets_are_present(self):
        for filename in (*SOUND_FILES.values(), MUSIC_MENU):
            with self.subTest(filename=filename):
                path = ASSET_DIR / filename
                self.assertTrue(path.is_file(), f"missing {path}")
                self.assertGreater(path.stat().st_size, 1024)

    def test_trim_silence_crops_leading_and_trailing_dead_air(self):
        # Mirrors the supplied web-swing.mp3: a burst buried after a long
        # silent lead-in, which would otherwise never be reached before the
        # next swing replaced it on the channel.
        rate = SAMPLE_RATE
        quiet = [0, 0] * rate                     # 1.0s of silence
        burst = [20_000, -20_000] * (rate // 5)   # 0.2s of signal
        raw = array("h", quiet + burst + quiet).tobytes()

        trimmed = trim_silence(raw)
        frames = len(trimmed) // (OUTPUT_CHANNELS * 2)
        duration = frames / rate

        self.assertLess(duration, 0.45, "dead air survived the trim")
        self.assertGreater(duration, 0.18, "trim ate the actual sound")

        # Audible within the first 60ms — the deliberate lead-in and de-click
        # fade account for roughly the first 20.
        head = array("h")
        head.frombytes(trimmed[: OUTPUT_CHANNELS * 2 * (rate * 60 // 1000)])
        self.assertTrue(any(head), "audible content should start immediately")

    def test_trim_silence_leaves_tight_and_silent_buffers_alone(self):
        tight = array("h", [15_000, -15_000] * (SAMPLE_RATE // 10)).tobytes()
        self.assertEqual(trim_silence(tight), tight)

        silent = array("h", [0, 0] * (SAMPLE_RATE // 10)).tobytes()
        self.assertEqual(trim_silence(silent), silent)

        self.assertEqual(trim_silence(b""), b"")

    def test_silent_stub_matches_the_real_sound_system_interface(self):
        # The 3D runner swaps in _SilentSoundSystem for --no-audio and headless
        # runs. Any public method added here that it lacks is an AttributeError
        # at runtime in exactly the configuration nobody tests interactively.
        from spidergame.render3d.game import _SilentSoundSystem

        expected = {
            name for name in vars(SoundSystem)
            if callable(getattr(SoundSystem, name)) and not name.startswith("_")
        }
        missing = expected - set(dir(_SilentSoundSystem))
        self.assertEqual(missing, set(), f"silent stub missing {missing}")

        stub = _SilentSoundSystem()
        args = {"handle": ((), swinging_sim()), "update": (swinging_sim(),)}
        for name in sorted(expected):
            with self.subTest(method=name):
                getattr(stub, name)(*args.get(name, ()))

    def test_impact_has_a_synthesised_fallback(self):
        bank = _build_pcm_bank()
        self.assertIn("impact", bank)
        samples = array("h")
        samples.frombytes(bank["impact"])
        self.assertTrue(any(samples), "impact fallback is silent")

    def test_stop_can_fade_or_stop_immediately(self):
        mixer = FakeMixer()
        audio = SoundSystem(mixer=mixer)

        # The impact channel is excluded by design — see play_fall().
        swing_channels = [mixer.channels[i] for i in
                          (SHOT_CHANNEL, CABLE_CHANNEL, MOTION_CHANNEL)]

        audio.stop(55)
        for channel in swing_channels:
            self.assertEqual(channel.fadeouts, [55])
        self.assertEqual(mixer.channels[IMPACT_CHANNEL].fadeouts, [])

        audio.stop(0)
        for channel in swing_channels:
            self.assertEqual(channel.stops, 1)
        self.assertEqual(mixer.channels[IMPACT_CHANNEL].stops, 0)

        # close() is the one path that must silence everything.
        audio.close()
        self.assertEqual(mixer.channels[IMPACT_CHANNEL].stops, 1)


if __name__ == "__main__":
    unittest.main()
