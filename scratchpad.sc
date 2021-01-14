~path = Document.current.dir ++ "/etude_3_seq.json";
~chords = File.open(~path, "r").readAllString;
~chords[0]