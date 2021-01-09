~path = Document.current.dir ++ "/chord_sequence.json";
~chords = File.open(~path, "r").readAllString.interpret;

~freq = 300;

(
~play = {arg freq, vol, size; {
	SinOsc.ar([freq, freq], 0, vol/size)
}.play}
);


~chords[3].do({arg freq; ~play.value(freq, 0.1, ~chords[0].size)});
~metric_dur = 5;


~rhythmic_sequence = {arg

~rands = Array.fill(~chords.size, {1.0.rand + 1});
~dur_seq = Array.fill(~chords.size, {arg i; var rhythm, sums, item, cutoff;
	rhythm = ~rands[i] * [0.2, 0.3, 0.22, 0.16, 0.11];
	'here'.postln;
	~rands[i].postln;
	rhythm = rhythm.wrapExtend(10);
	sums = Array.fill(rhythm.size, {arg i; rhythm[..i+1].sum});
	sums.postln;
	item = sums.detect({arg it, i; it > ~metric_dur});
	cutoff = sums.indexOf(item);
	rhythm = rhythm[..cutoff];
	rhythm = rhythm.normalizeSum * ~metric_dur;
});
~prand_sequence = Array.fill(~chords.size, {arg i; Prand(~chords[i], ~dur_seq[i].size)});

((
/*TempoClock.default.temp = 84/60;*/
p = Pbind(
	\freq, Pseq(~chords, inf),
	\dur, ~metric_dur,
);
p.play;
);

(
b = Pbind(
	\freq, Pseq(~prand_sequence, inf),
	\dur, Pseq(~dur_seq.flat, inf),
);
b.play;
))

//
//
//
// List[1, 2, 3, 4].detect({ arg item, i; item.even });
//
// p.stop;
//
// a = [0, 1, 2, 3]
//
// (
// var a, x;
// a = Pfunc({arg i; ~chords[i] });
// a.asStream
// x = a.asStream;
// x.nextN(12).postln;
// x.reset;
// )
//
// block {|break|
// 	100.do {|i|
// 		i.postln;
// 		if (i == 7) { break.value(999) }
// 	};
// }