~path = Document.current.dir ++ "/etude_3_seq.json";
~chords = File.open(~path, "r").readAllString.interpret;
~get_amp = {arg partials, falloff; var out;
		out = Array.fill(partials.size, {arg i;
			falloff ** partials[i]});
		out / out.sum;
	};

~buffs = Buffer.allocConsecutive(24, s, 512*4);
~buffs.do({arg item, i; item.sine2(~chords[i], ~get_amp.value(~chords[i], 0.90))});


// slide between
({var a, trig, env, bufstart;
	a = Dbrown.new(0, 22.99, 0.5);
	bufstart = 90;
	env = EnvGen.kr(Env([bufstart, bufstart + 22.9], [20]));
	VOsc.ar(env, 100, 0, 0.25)*[1, 1];
}.play)


// step through
({var a, trig, env, bufstart, vals, durs;
	// a = Dbrown.new(0, 22.99, 0.5);
	bufstart = 90;
	vals = 0.99 * Array.series(23, bufstart, 1).stutter(2);
	durs = [0.125, 0].wrapExtend(vals.size);
	env = EnvGen.kr(Env(vals, durs));
	VOsc.ar(env, 100, 0, 0.25)*[1, 1];
}.play)

[1, 2, 3] * 2

({var bufstart = 90;
	VOsc.ar(bufstart, 100, 0, 0.25)*[1, 1];
}.play)

