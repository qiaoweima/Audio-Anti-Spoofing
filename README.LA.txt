=====================================================================================================
ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database

Logical access (LA)
=====================================================================================================


1. Directory Structure
_______________________

  --> LA  
          --> ASVspoof2019_LA_asv_protocols
          --> ASVspoof2019_LA_asv_scores
	  --> ASVspoof2019_LA_cm_protocols
          --> ASVspoof2019_LA_dev
          --> ASVspoof2019_LA_eval
	  --> ASVspoof2019_LA_train
	  --> README.LA.txt


2. Description of the audio files
_________________________________

   ASVspoof2019_LA_train, ASVspoof2019_LA_dev, and ASVspoof2019_LA_eval contain audio files for training, development, and evaluation
   (LA_T_*.flac, LA_D_*.flac, and LA_E_*.flac, respectively). ASVspoof2019_PA_dev, and ASVspoof2019_PA_eval contain audio files to enroll ASV system. The audio files in the directories are in the flac format. 
   The sampling rate is 16 kHz, and stored in 16-bit.


3. Description of the protocols
_______________________________

CM protocols:

   ASVspoof2019_LA_cm_protocols contains protocol files in ASCII format for ASVspoof countermeasures:

   ASVspoof2019.LA.cm.train.trn.txt: training file list
   ASVspoof2019.LA.cm.dev.trl.txt: development trials
   ASVspoof2019.LA.cm.eval.trl.txt: evaluation trials 
	
   Each column of the protocol is formatted as:
   
   SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY

   	1) SPEAKER_ID: 		LA_****, a 4-digit speaker ID
   	2) AUDIO_FILE_NAME: 	LA_****, name of the audio file
   	3) SYSTEM_ID: 		ID of the speech spoofing system (A01 - A19),  or, for bonafide speech SYSTEM-ID is left blank ('-')
   	4) -: 			This column is NOT used for LA.
	5) KEY: 		'bonafide' for genuine speech, or, 'spoof' for spoofing speech

   Note that: 
   
   	1) the third column is left blank (-) to make the structure coherent with physical access file list;
   	2) Brief description on LA spoofing systems, where TTS and VC denote text-to-speech and voice-conversion systems:
   	
        A01	TTS	neural waveform model
        A02	TTS	vocoder
        A03	TTS	vocoder
        A04	TTS	waveform concatenation
        A05	VC	vocoder
        A06	VC	spectral filtering
        		
        A07	TTS	vocoder+GAN
        A08	TTS	neural waveform
        A09	TTS	vocoder
        A10	TTS	neural waveform
        A11	TTS	griffin lim
        A12	TTS	neural waveform
        A13	TTS_VC	waveform concatenation+waveform filtering
        A14	TTS_VC	vocoder
        A15	TTS_VC	neural waveform
        A16	TTS	waveform concatenation
        A17	VC	waveform filtering
        A18	VC	vocoder
        A19	VC	spectral filtering
   
ASV protocols:
   
   ASVspoof2019_LA_asv_protocols contains the protocol files for ASV system

	ASVspoof2019.LA.asv.<1>.<2>.<3>.txt
	where
	<1> is either 'dev' or 'eval' based on whether the files describe the development or evaluation protocol,
	<2> ie either 'male (m)' or 'female (f)' separating the genders from each other or 'gender independent (gi)' 
	    contains trials for both genders (male trials followed by female trials),
	<3> is either 'trl' or 'trn' (trl = trial list, trn = speaker enrollment list).

	Trial (trl) file format for LA scenario:
	1st column: claimed speaker ID
	2nd column: test file ID
	3rd column: spoof attack ID (or 'bonafide' if the speech is not spoofed)
	4th column: key (target = target trial, nontarget = impostor trial, spoof = spoofing attack)

	Enrollment (trn) file format:
	1st column: ID of enrolled speaker
	2nd column: IDs of files used in the enrollment separated by commas
	

4. Baseline ASV scores
______________________

   ASVspoof2019_LA_asv_scores contains the scores calculated by a baseline ASV system for t-DCF evaluation
   
   	ASVspoof2019.LA.asv.dev.gi.trl.scores.txt:  scores given by the ASV system for development set data
   	ASVspoof2019.LA.asv.eval.gi.trl.scores.txt: scores given by the ASV system for evaluation set data
   	
   Each column is formatted as:
   
   CM_KEY ASV_KEY SCORES

   	1) CM_KEY: 		'bonafide' for genuine speech, or, the ID of the spoofing attack (A01 - A19)
   	2) ASV_KEY: 		'target' for claimed speaker, or, 'nontarget' for impostor speaker, or, 'spoof' for spoofing speech
   	3) SCORES: 		similarity score value

-------------------------------------------------------------------------------------------------------------------------------------

