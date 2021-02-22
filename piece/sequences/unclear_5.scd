~fund = 200

(
SynthDef(\inst, {|out, freq, amp = 0.1, gate = 1 |
	var audio = Blip.ar(freq, 16, amp);
	var env = EnvGen.kr(Env.adsr(), gate, doneAction: 2);
	var filterEnv = EnvGen.kr( Env.new([1, 0.6, 0], [0.2, 0.2], 'exp', 1));
	var filter = MoogFF.ar(audio * env, freq  * (2 ** (6 * filterEnv)), 2);
	Out.ar(0, [1, 1] * filter);
}).add;
)

Filter

(
b = Pbind(
	\instrument, \inst,
	\freq, Prand(~fund * [1, 9/8, (9/8)**2], inf),
	\dur, Prand([0.4, 0.6, 0.5], inf),
	\amp, 0.5,
);
b.play;
)

