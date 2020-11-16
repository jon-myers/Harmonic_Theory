\version "2.20.0"   %! abjad.LilyPondFile._get_format_pieces()
\language "english" %! abjad.LilyPondFile._get_format_pieces()

\header { %! abjad.LilyPondFile._get_formatted_blocks()
    tagline = ##f
} %! abjad.LilyPondFile._get_formatted_blocks()

\layout {}

\paper {}

\score { %! abjad.LilyPondFile._get_formatted_blocks()
    \new PianoStaff
    <<
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            \once \override NoteHead.color = #(x11-color 'black)
            \time 6/4
            c'4
            s4
            \once \override NoteHead.color = #(x11-color 'MediumSeaGreen)
            ef'4
            s4
            \once \override NoteHead.color = #(x11-color 'DarkOrchid)
            e'4
            \once \override NoteHead.color = #(x11-color 'RoyalBlue)
            b'4
        }
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            \clef "bass"
            s4
            \once \override NoteHead.color = #(x11-color 'SaddleBrown)
            f4
            ^ \markup {
                \raise
                    #1.5
                    \concat
                        {
                            \tiny
                                -2
                            \hspace
                                #1.7
                            \tiny
                                -33
                            \hspace
                                #1.7
                            \tiny
                                +31
                            \hspace
                                #1.7
                            \tiny
                                -14
                            \hspace
                                #1.7
                            \tiny
                                -12
                        }
                }
            s4
            \once \override NoteHead.color = #(x11-color 'red)
            d4
            s4
            s4
        }
    >>
} %! abjad.LilyPondFile._get_formatted_blocks()

\layout { %! abjad.LilyPondFile._get_formatted_blocks()
    \context {
        \Score
        \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1 12)
    }
} %! abjad.LilyPondFile._get_formatted_blocks()