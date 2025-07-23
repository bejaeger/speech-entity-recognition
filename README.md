# speech-entity-recognition

Sketch of a method to enhance spoken named entity recognition in speech recognition pipelines

## Time allocation
- 1h: Reading and researching problem
- 1h: Sketching out possible solutions
- 3h: Implementing ContextEncoder and Transducer model
- (2h: Implementing entire training pipeline including all supporting code. Not asked for but was a good way to test the model. This is in parentheses because it wasn't part of the deliverables. I did it because it was fun :) )

## Proposed solution to the problem

The code implements a transducer model (based on an LSTM audio encoder and a transformer-based prediction network) that
is biased towards a set of vocabulary encoded with a context encoder. This makes the model learn to recognize the entities in the context. To "test" this method, a very naive training pipeline is implemented that overfits on the audio provided in the assignment. The model outputs "calinovski" if "calinovski" is in the context/list of entities and "kalinowski" if "kalinowski" is in the context/list of entities. This is a very rough sketch and far from being a final solution. For example, it's likely that the model trained below just overfits with the context encoder and memorizes the sentence.

## Deliverable

### Answers

#### How could we improve on the limitations of the current approach?
An improvement could be to find a way to directly bias the model (in an E2E fashion) to a set of entities (also called "context" in the following), instead of a multi-stage pipeline. This could be a context encoder which output can be attended to. 
Another idea would be to use a sliding window approach and run the CLAP model in parallel to the ASR pipeline. This, however, would come with significant increase in compute requirements.
Maybe (and this is just me speculating without doing further research) a CLAP model could be baked into the ASR pipeline and used in a streaming fashion.

#### What could a be completely new approach to solve this problem?
There are two main ideas I considered implementing:
- Shallow fusion: Boosting certain paths during decoding, based on a list of entities/context. This is the traditional way to improve transcription of unknown entities or jargon. Upon research, this generally seems to work well but has limitations in terms of accuracy, especially going to longer external contexts.
- Deep context biasing: Adding a context encoder to the transducer model and learn to attend to a set of vocabularies/context. 

I decided to implement the second idea because it's a more modern approach.

### Code
The main deliverable was to implement a sketch of a pyTorch model of a suggested architecture. This can be found in `src/model.py` (ContextEncoder and the way it is used in the Transducer model).
The supporting code to run the training pipeline is kept separate and can be found in `src/supporting_code/` and `train.py`.

## Possible extensions / ideas

Some ideas for extensions to the current solution:
- The cross attention to the encoded context could be added at different stages in the transducer model, e.g. after the audio encoder and before the joint network.
- The context encoder could make use of pretrained models, such as embedding models like [Multilingual-E5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- The context could also be enriched with a more detailed description. For example instead of just adding the words "Herr Kalinowski" one could add something like "Type: Name of person, Prefixes: Herr, Content: Kalinowski". A pretrained transformer would then encode this into a rich representation that the ASR model can make better use of.

## Tool Use

I heavily made use of Cursor and AI to implement the solution, mostly to get a quick implementation of the entire training pipeline to test the method.

## Run code

### Install
Install dependencies with:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Tests
Run tests with:
```bash
pytest test/
```

### Training
Run training with:
```bash
python train.py
```
After about 700 epochs this will produce a transcription that entails "calinovski", if "calinovski" is in the context/list of entities and "kalinowski", if "kalinowski" is in the context/list of entities.
