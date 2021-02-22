~path = Document.current.dir ++ "/melody_0.json";
~melodies = File.open(~path, "r").readAllString.interpret;

b = Pgauss(length: inf)
~rands = 0.25 * (2 ** b.asStream.nextN(500));
(
a = Pbind(
	\freq, Pseq(~melodies[0], inf),
	\dur, Pseq(~rands, inf),
	\pan, -1
).play
);
a.play;
a.stop;
~melodies[0]
~melodies[1]
(
b = Pbind(
	\freq, Pseq(~melodies[1], inf),
	\dur, Pseq(~rands, inf),
	\pan, 0
).play);
b.play;
b.stop


(
c = Pbind(
	\freq, Pseq(~melodies[2], inf),
	\dur, Pseq(~rands, inf),
	\pan, 0
).play);
c.play;
c.stop;

(
d = Pbind(
	\freq, Pseq(~melodies[3], inf),
	\dur, Pseq(~rands, inf),
	\pan, -1
).play);
d.play;
d.stop;

(
e = Pbind(
	\freq, Pseq(~melodies[4], inf),
	\dur, Pseq(~rands, inf),
	\pan, 1
).play);
e.play;
e.stop

(
f = Pbind(
	\freq, Pseq(~melodies[5], inf),
	\dur, Pseq(~rands, inf),
	\pan, -1
).play);
f.play;
f.stop;

(
g = Pbind(
	\freq, Pseq(~melodies[6], inf),
	\dur, Pseq(~rands, inf),
	\pan, 1
).play);
g.play;
g.stop

(
h = Pbind(
	\freq, Pseq(~melodies[7], inf),
	\dur, Pseq(~rands, inf),
	\pan, -1
).play);
h.play;
h.stop