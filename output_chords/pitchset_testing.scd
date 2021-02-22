~fund = 100;
~trip = {arg a; a ++ (2*a) ++ (4*a)};

~chords = File.open("/Users/Jon/Documents/2020/harmony-figures/output_chords/ratios.json", "r");
~chords = ~chords.readAllString.parseYAML;
~chords = Array.fill(~chords.size, {arg i; ~chords[i].asFloat});



(~dc_alg = {arg choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0;
	var selections = [], weight_array, sum, probs;
	if (weights ==0) {weights = Array.fill(choices.size, {1})};
	if (counts == 0) {counts = Array.fill(choices.size, {1})};
	epochs.do({var selection_index;
		weight_array = Array.fill(choices.size, {arg i; weights[i] * (counts[i]**alpha)});
		probs = Array.fill(choices.size, {arg i; weights[i] * (counts[i]**alpha) / weight_array.sum});
		selection_index = Array.series(choices.size).wchoose(probs);
		counts = Array.fill(counts.size, {arg i; counts[i] + 1}).put(selection_index,0);
		selections = selections ++ [choices[selection_index]];
	});
	selections;
});

~chords[0]
(
~chord_index = 7;
~perm_index = 0;
p = Pbind(
	\freq, Pseq(~dc_alg.value(~trip.value(~chords[6 * ~chord_index + ~perm_index]) * ~fund, 1000), inf),
	\dur, Prand([0.05, 0.08, 0.09, 0.1,0.15,0.2,0.25,0.3] ,inf),
	\vol, Prand([0.25, 0.4, 0.65], inf)
).play;

);
~chords[1]
~chords[0]

6 * ~chord_index + ~perm_index
~chords[31]