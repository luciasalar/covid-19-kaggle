
class MatchArticleBody:
    def __init__(self, path, selected_id):
        """Define varibles."""
        self.path = path
        self.selected_id = selected_id


    def read_folder(self):
        """
        Creates a nested dictionary that represents the folder structure of rootdir
        """
        rootdir = self.path.rstrip(os.sep)

        article_dict = {}
        for path, dirs, files in os.walk(rootdir):
            for f in files:
                file_id = f.split('.')[0]
                #print(file_id)
                try:
                # load json file according to id
                    with open(self.path + f) as f:
                        data = json.load(f)
                except:
                    pass
                article_dict[file_id] = data

        return article_dict


    def extract_bodytext(self):
        """Unpack nested dictionary and extract body of the article"""
        body = {}
        article_dict = self.read_folder()
        for k, v in article_dict.items():
            strings = ''
            prevString = ''
            for entry in v['body_text']:
                strings = strings + prevString
                prevString = entry['text']

            body[k] = strings
        return body


    def get_title_by_bodykv(self, article_dict, keyword):
        """Search keyword in article body and return title"""

        article_dict = self.read_folder()
        selected_id = self.extract_id_list()

        result = {}
        for k, v in article_dict.items():
            for entry in v['body_text']:
                if (keyword in entry['text'].split()) and (k in selected_id):
                    result[k] = v['metadata']['title']

        return result


    def extract_id_list(self):
        """Extract ids from the selected text. """
        selected_id = []
        for k, v in self.selected_id.items():
            selected_id.append(str(v['sha']).split(';')[0])
            try:
                selected_id.append(str(v['sha']).split(';')[1])
                selected_id.append(str(v['sha']).split(';')[2])
                selected_id.append(str(v['sha']).split(';')[3])
            except:
                pass

        return selected_id


    def select_text_w_id(self):
        body_text = self.extract_bodytext()
        selected_id = self.extract_id_list()
        selected_text = {}
        for k, v in body_text.items():
            if k in selected_id:
                selected_text[k] = v
        return selected_text