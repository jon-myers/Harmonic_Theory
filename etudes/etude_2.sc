~path = Document.current.dir ++ "/etude_2_seq.json";
~chords = File.open(~path, "r").readAllString.interpret;

~freq = 250;
~metric_dur = 0.5;

(
/*TempoClock.default.temp = 84/60;*/
p = Pbind(
	\freq, Pseq(~chords, inf),
	\dur, Prand([0.12, 0.22,0.32, 0.42, 0.52, 0.62, 0.72], inf)
	\amp, 0.1,
);
p.play;
);