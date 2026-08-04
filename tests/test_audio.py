import unittest
from array import array
from types import SimpleNamespace

import pygame

from spidergame.audio import (
    CABLE_CHANNEL,
    MOTION_CHANNEL,
    OUTPUT_CHANNELS,
    SAMPLE_RATE,
    SHOT_CHANNEL,
    SoundSystem,
    _build_pcm_bank,
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

        self.assertEqual(first.keys(), {"shot", "attach", "miss", "release", "whoosh"})
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
        self.assertEqual(mixer.num_channels, 3)
        self.assertEqual(mixer.reserved, 3)
        self.assertEqual(set(mixer.channels), {SHOT_CHANNEL, CABLE_CHANNEL, MOTION_CHANNEL})

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

    def test_stop_can_fade_or_stop_immediately(self):
        mixer = FakeMixer()
        audio = SoundSystem(mixer=mixer)

        audio.stop(55)
        for channel in mixer.channels.values():
            self.assertEqual(channel.fadeouts, [55])
        audio.stop(0)
        for channel in mixer.channels.values():
            self.assertEqual(channel.stops, 1)


if __name__ == "__main__":
    unittest.main()
