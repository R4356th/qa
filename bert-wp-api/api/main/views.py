from django.http import HttpResponse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch import cat, argmax, masked_select, split, tensor, squeeze, unsqueeze
from wikipedia import search, page
from collections import OrderedDict
from logging import getLogger, ERROR
getLogger("transformers.tokenization_utils").setLevel(ERROR)

# Create your views here.
def index(request):
    return HttpResponse('Wrong route! This API has been made by Radman Siddiki for working with BERT and the index does not serve any purpose here.')

def get_answer(request):
    reader = DocumentReader("deepset/bert-base-cased-squad2") 
    param = request.GET.get('q', '')
    results = search(param)
    text = page(results[0], auto_suggest=False).content
    reader.tokenize(param, text)
    answer = reader.get_answer()
    return HttpResponse(answer)

class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='bert-large-uncased'):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        self.inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1

        chunked_input = OrderedDict()
        for k,v in self.inputs.items():
            q = masked_select(v, qmask)
            c = masked_select(v, ~qmask)
            chunks = split(c, chunk_size)
            
            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = cat((q, chunk))
                if i != len(chunks)-1:
                    if k == 'input_ids':
                        thing = cat((thing, tensor([102])))
                    else:
                        thing = cat((thing, tensor([1])))

                chunked_input[i][k] = unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(**chunk, return_dict=False)

                answer_start = argmax(answer_start_scores)
                answer_end = argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + " / "
            return answer
        else:
            answer_start_scores, answer_end_scores = self.model(**self.inputs, return_dict=False)

            answer_start = argmax(answer_start_scores) 
            answer_end = argmax(answer_end_scores) + 1
        
            return self.convert_ids_to_string(self.inputs['input_ids'][0][
                                              answer_start:answer_end])

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))
