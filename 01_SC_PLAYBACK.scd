(
var
numIter = 18,
prevVal = -1,
directory,
bufferLength,
minBlue, maxBlue,
minGreen, maxGreen,
minRed, maxRed,
playSynth,
currentIter = 0,
ampGrain,
ampNorm,
semData = Semaphore.new(0),
osc = NetAddr("127.0.0.1", 57121),
startTime;

// Create a mix bus for the entire output
~mixBus = Bus.audio(s, 2); // Stereo mix bus

ampGrain = (1/numIter) * (currentIter+1);
ampNorm = 1-ampGrain;


// SynthDef for the main playback
SynthDef(\playAudio, { |monoBuf, stereoBuf, rate = 1, amp1 = 0.1, amp2 = 0.1, pan = 0, start = 0|
	var orig, lpf, sig, gran;

	// Play original stereo file
	orig = PlayBuf.ar(2, stereoBuf, rate, startPos: start, loop: 1);
	orig = orig * amp1;

	// Grain synthesis
	gran = GrainBuf.ar(
		numChannels: 2,
		trigger: Dust.ar(\grainFreq.kr(40)),
		dur: \grainDur.kr(1),
		sndbuf: monoBuf,
		rate: \grainRate.kr(1),
		pos: Line.kr(start, 1, BufFrames.kr(monoBuf) - start / SampleRate.ir)
	);

	gran = LPF.ar(gran, Lag.kr(\filterFreq.kr(400)), 1);
	gran = gran * \volumeAmp.kr(1);
	gran = gran * amp2;

	sig = Mix([orig, gran]);
	sig = sig * 0.5 * \fadeIn.kr(1) * \fadeOut.kr(1);


	Out.ar(~mixBus.index, sig);
}).add;

// Reverb SynthDef applied to the mix bus
SynthDef(\reverbEffect, { |in, out, roomWetness = 0.75, reverbDecay = 0.5|
	var sound, wet, dry, processed;

	// Input from mix bus
	sound = In.ar(in, 2);
	wet = FreeVerb.ar(sound, roomWetness, reverbDecay, 0.9); // Reverb effect
	dry = sound * (1 - roomWetness); // Combine with dry signal
	processed = wet + dry;

	// Output to main speakers
	Out.ar(out, processed);
}).add;



// OSCdef for evaluating blocks
OSCdef(\evaluateBlock, { |msg|
	// Reset parameters before processing new data
	~prevAudioFile = nil;
	~prevMonoAudioFile = nil;
	~curAudioFile = nil;
	~curMonoAudioFile = nil;
	~redData = nil;
	~greenData = nil;
	~blueData = nil;

	// Optionally reset other parameters if needed
	minRed = 0; maxRed = 1;
	minBlue = 0; maxBlue = 1;
	minGreen = 0; maxGreen = 1;
	ampGrain = 0; ampNorm = 1;
	currentIter = 0; prevVal = -1;

	// Post reset status
	"Parameters reset.".postln;

	// Free existing OSCdefs to remove previous listeners
	OSCdef(\redData, nil);
	OSCdef(\greenData, nil);
	OSCdef(\blueData, nil);

	// Re-create OSCdef listeners
	(
		OSCdef(\redData, { |msg|
			~redData = msg[1..];
			"Red channel received".postln;
			minRed = ~redData.minItem;
			maxRed = ~redData.maxItem;
			semData.signal;
		}, "/image/red", recvPort: 57120);

		OSCdef(\greenData, { |msg|
			~greenData = msg[1..];
			"Green channel received".postln;
			minGreen = ~greenData.minItem;
			maxGreen = ~greenData.maxItem;
		}, "/image/green", recvPort: 57120);

		OSCdef(\blueData, { |msg|
			~blueData = msg[1..];
			"Blue channel received".postln;
			minBlue = ~blueData.minItem;
			maxBlue = ~blueData.maxItem;
		}, "/image/blue", recvPort: 57120);
	);

	osc.sendMsg("/trigger/evaluate");

	"Code block evaluated".postln;

	Routine{
		var sem = Semaphore.new(0);
		var lastValue = nil,
		line, currentFile, prevFile, ampGrain, ampNorm, index, otherIndex, bufFrames, task;

		semData.wait;

		directory ="/Users/linusziegler/Documents/CODE/SEM7/AICA/repo/AICA" ++ "/" ++(1..1).choose++"/" ;


		~curAudioFile = Buffer.read(s, directory++"0.mp3",  action: {sem.signal});
		~curMonoAudioFile = Buffer.readChannel(s, directory++"0.mp3", channels: [0]);


		bufFrames = ~curAudioFile.numFrames;

		sem.wait;
		~curAudioFile.postln;

		bufferLength = ~curAudioFile.duration;
		"ELINDULT".postln;



		~synths = [nil, nil, nil]; // Array to store up to 2 active synths
		~lastValue = nil;

		ampGrain = (1/numIter) * (0);
		ampNorm = 1 - ampGrain;

		currentFile = (directory ++ "0.mp3");
		currentFile.postln;

		~curAudioFile = Buffer.read(s, currentFile);
		~curMonoAudioFile = Buffer.readChannel(s, currentFile);

		~reverbSynth = Synth(\reverbEffect, [
			\in, ~mixBus.index,  // Input from mix bus
			\out, 0,             // Output to speakers
			\roomWetness, 0.75,  // Adjust room wetness
			\reverbDecay, 0.5    // Adjust decay
		]);


		(
			~synths[2] = Synth(\playAudio, [
				amp1: 1,
				amp2: 0,
				stereoBuf: ~curAudioFile.bufnum,
				monoBuf: ~curMonoAudioFile.bufnum,
				grainFreq: 1 / (bufferLength / ~blueData.size),
				start: 0
			]);
		);

		~synths[2].set(\fadeIn, 1);

		startTime = SystemClock.seconds;

		~task = Task {
			loop {
				startTime = SystemClock.seconds;

				if(~lastValue.notNil) {~synths.[0].set(\start, 0); ~synths.[1].set(\start, 0)};

				("START TIME: " ++ startTime).postln;

				bufferLength.wait;
			}
		};

		~task.start;


		Routine {
			~redData.do { |in|
				var out = in.linlin(minRed, maxRed, 20, 20000);
				~synths.[0].set(\filterFreq, out);
				~synths.[1].set(\filterFreq, out);
				(bufferLength / ~redData.size).wait;
			};
		}.play;

		Routine {
			~blueData.do { |in|
				var out = in.linlin(minBlue, maxBlue, 0.125, 8);
				~synths.[0].set(\grainRate, out, \grainDur, out);
				~synths.[1].set(\grainRate, out, \grainDur, out);
				(bufferLength / ~blueData.size).wait;
			};
		}.play;

		Routine {
			~greenData.do { |in|
				var out = in.linlin(minGreen, maxGreen, 0.05, 1);
				~synths.[0].set(\volumeAmp, out);
				~synths.[1].set(\volumeAmp, out);
				(bufferLength / ~greenData.size).wait;
			};
		}.play;


		OSCdef(\knobValue, { |msg|
			var value = msg[1].asInteger,
			fade = msg[2],
			elapsedTime,
			startFrame;

			if ((value != ~lastValue).and(~lastValue.notNil)) {  // Check if the current value is different from the last value
				"Received knob value: ".format(value).postln;

				~lastValue = value;  // Update lastValue

				ampGrain = (1/numIter) * (value);
				ampNorm = 1 - ampGrain;

				currentFile = (directory ++ value ++ ".wav");
				prevFile = (directory ++ (value-1) ++ ".wav");

				~prevAudioFile = Buffer.read(s, currentFile);
				~prevMonoAudioFile = Buffer.readChannel(s, currentFile);
				~curAudioFile = Buffer.read(s, currentFile);
				~curMonoAudioFile = Buffer.readChannel(s, currentFile);


				if(~synths[0].notNil) {~synths[0].free};
				if(~synths[1].notNil) {~synths[1].free};
				if(~synths[2].notNil) {~synths[2].free; ~synths[2] = nil};

				elapsedTime = SystemClock.seconds-startTime;

				("ELAPSED TIME: " ++ elapsedTime).postln;

				startFrame = elapsedTime * s.sampleRate;

				// Create a new synth and store it in the current index
				~synths[0] = Synth(\playAudio, [
					amp1: ampNorm,
					amp2: ampGrain,
					stereoBuf: ~curAudioFile.bufnum,
					monoBuf: ~curMonoAudioFile.bufnum,
					grainFreq: 1 / (bufferLength / ~blueData.size),
					start: startFrame
				]);

				~synths[1] = Synth(\playAudio, [
					amp1: ampNorm,
					amp2: ampGrain,
					stereoBuf: ~prevAudioFile.bufnum,
					monoBuf: ~prevMonoAudioFile.bufnum,
					grainFreq: 1 / (bufferLength / ~blueData.size),
					start: startFrame
				]);

			};



			~synths[0].set(\fadeIn, fade);
			~synths[1].set(\fadeOut, 1-fade);



		}, '/knob', recvPort: 57122);

		~lastValue = 0;

	}.play;

	~lastValue = nil;

}, "/evaluate/code", recvPort: 57120);


// OSCdef to handle stop and reset from Python
OSCdef(\stopAndReset, { |msg, time, addr, recvPort|

	~synths.do{
		arg obj, index;
		if(obj.notNil){obj.free; ~synths[index] = nil};
	};

	~reverbSynth.free;
	~reverbSynth = nil;

	// Reset any global variables or states as needed
	~prevAudioFile = nil;
	~prevMonoAudioFile = nil;
	~curAudioFile = nil;
	~curMonoAudioFile = nil;
	~redData = nil;
	~greenData = nil;
	~blueData = nil;

	// Reset other parameters if needed
	minRed = 0; maxRed = 1;
	minBlue = 0; maxBlue = 1;
	minGreen = 0; maxGreen = 1;
	ampGrain = 0; ampNorm = 1;
	currentIter = 0; prevVal = -1;

	~task.stop;

	// Post reset confirmation
	"All synths stopped and parameters reset.".postln;
}, "/evaluate/code", recvPort: 57123);

)


