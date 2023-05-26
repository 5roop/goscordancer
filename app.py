from pydub import AudioSegment
from pathlib import Path
import pandas as pd
import streamlit as st
import json
import datetime

st.set_page_config(layout="wide")


problem_log_path = Path("/ssd/peter/problems.jsonl")
alignments_ctm_path = Path("/ssd/peter/alignments.ctm")
data_csv_path = Path("/ssd/peter/001_data.csv")
audio_dir = Path("/ssd/peter/final")


def flush_note():
    st.session_state.note = ""


st.header("🪿❤️💃")
st.header("GOScordancer")
if not alignments_ctm_path.exists():
    st.error("Alignments file doesn't exist!")
if not data_csv_path.exists():
    st.error("Data path doesn't exist!")
if not audio_dir.exists():
    st.error("Audio dir doesn't exist!")
audio_paths = list(audio_dir.glob("*.wav"))
if (l := len(audio_paths)) == 0:
    st.error("Audio dir is empty")
else:
    st.write(f"Found  {l:,} audio files")


@st.cache_data
def load_alignments():
    df =  pd.read_table(
        alignments_ctm_path,
        delim_whitespace=True,
        names="segment channels start duration word confidence".split(),
    )
    df["word"] = df.word.fillna("")
    return df

with st.spinner("Loading alignments..."):
    alignments = load_alignments()


@st.cache_data
def load_texts():
    df =  pd.read_csv(data_csv_path)
    df["text"] = df.text.fillna("")
    return df


@st.cache_resource
def get_random_segment():
    return alignments.segment.sample(1).values[0]


chosen_segment = get_random_segment()
with st.spinner("Loading transcripts..."):
    texts = load_texts()


with st.form("mode_choosing"):
    st.write(
        "Pick a mode. Random segment will be drawn every time the 'Choose' button is clicked."
    )
    mode = st.radio("Mode:", ["inspect a random segment", "find a segment with RegEx", "find a word in transcripts"])
    if "random" in mode:
        mode = "random"
    elif "find a segment" in mode:
        mode = "findseg"
    elif "find a word" in mode:
        mode = "findword"
    else:
        raise NotImplementedError(f"Option {mode} is not recognised")
    st.form_submit_button(label="Choose!", on_click=get_random_segment.clear)

if mode == "findseg":
    with st.form("regex_entering"):
        query = st.text_input("Regex to filter recordings\n(e.g. `.*P701113.*`)")
        st.form_submit_button()
    matches = alignments.segment.str.match(query)
    st.write(f"Found {matches.sum():,} segments.")
    alignments_subset = alignments.loc[matches, :]
    texts_subset = texts.loc[texts.id.str.match(query), :]
    st.dataframe(texts_subset["id text".split()], use_container_width=True)
    with st.form("segment choosening"):
        chosen_segment = st.selectbox("Choose a segment:", texts_subset["id"].tolist())
        st.form_submit_button()
elif mode == "random":
    alignments_subset = alignments.loc[alignments.segment == chosen_segment, :]
    st.write("Kaldi output:")
    st.dataframe(alignments_subset, use_container_width=True)
    texts_subset = texts.loc[texts.id == chosen_segment, :]
    st.write("TEI transcript:")
    st.dataframe(texts_subset)
elif mode == "findword":
    with st.form("regex_entering"):
        query = st.text_input(
            "RegEx to filter transcripts\n(e.g. `krava`). Query will be nested between `^.*` and `.*$`",
            value="[dD]imnik"
            )
        query = "^.*"+query+".*$"
        st.form_submit_button()
    matches = alignments.word.str.match(query)
    st.write(f"Found {matches.sum():,} segments.")
    alignments_subset = alignments.loc[matches, :]
    texts_subset = texts.loc[texts.text.str.match(query), :]
    st.dataframe(texts_subset["id text".split()], use_container_width=True)
    with st.form("segment choosening"):
        chosen_segment = st.selectbox("Choose a segment:", texts_subset["id"].tolist())
        st.form_submit_button()
else:
    pass

with st.form("word choosening"):
    st.write(f"Chosen segment: {chosen_segment}")
    alignments_sub_subset = alignments_subset[
        alignments_subset.segment == chosen_segment
    ]
    words = alignments_sub_subset.word.tolist()
    ii = alignments_sub_subset.index.tolist()
    keys = {f"{i}: {w}": i for i, w in zip(ii, words)}
    key = st.radio("Choose a word:", keys.keys())
    alignment_row = alignments_sub_subset.loc[keys[key], :]
    st.form_submit_button(on_click=flush_note, label="Confirm selection")


texts_row = texts[texts.id == chosen_segment]
assert (
    texts_row.shape[0] == 1
), f"chosen segment filtering returns non-unique rows {row}"
segment_filename = texts_row["segment_filename"].tolist()[0]
audio_files_eligible = [i for i in audio_dir.glob("*.*") if segment_filename in str(i)]
num_found_files = len(audio_files_eligible)
if num_found_files < 1:
    raise FileNotFoundError("Audio file was not found")
elif num_found_files > 1:
    raise LookupError("Found more than one file!")
audio_file = audio_files_eligible[0]


c1, c2, c3, c4 = st.columns(4)
with st.spinner("Prepping audio files..."):
    input_audio_path = audio_dir / audio_file
    input_audio = AudioSegment.from_wav(str(input_audio_path))
    start = int(1000 * alignment_row["start"])
    end = start + int(1000 * alignment_row["duration"])
    path1 = Path("/ssd/peter/app/file1.wav")
    path2 = Path("/ssd/peter/app/file2.wav")
    path3 = Path("/ssd/peter/app/file3.wav")
    L = len(input_audio)
    input_audio[start:end].export(str(path1), format="wav")
    input_audio[max(start - 500, 0) : min(end + 500, L)].export(
        str(path2), format="wav"
    )
    (input_audio[start:end] + input_audio[start:end] + input_audio[start:end]).export(
        str(path3), format="wav"
    )
with c1:
    st.write("Word")
    st.audio(str(path1))
with c2:
    st.write("Word with 0.5s context")
    st.audio(str(path2))
with c3:
    st.write("Word, repeated 3 times")
    st.audio(str(path3))
with c4:
    st.write("Whole segment")
    st.audio(str(input_audio_path))


def report():
    problematic_segment = chosen_segment
    problematic_word = alignment_row["word"]
    string_to_dump = json.dumps(
        {
            "segment": problematic_segment,
            "audio_file": str(audio_file.name),
            "problematic_word": problematic_word,
            "start": f"{start/1e3:0.3f}",
            "end": f"{end/1e3:0.3f}",
            "description": st.session_state.note,
            "datetime_of_report": datetime.datetime.now().isoformat(),
        }
    )
    with open(str(problem_log_path), "a") as f:
        f.write(string_to_dump + "\n")


with st.form("problem reporting"):
    st.write("If you spot a problem, report it here:")
    problematic_segment = chosen_segment
    problematic_word = alignment_row["word"]
    note = st.text_input("Issue description", placeholder="", key="note")
    if st.form_submit_button(label="Report", on_click=report):
        st.info("Thanks for reporting!", icon="🪿")
