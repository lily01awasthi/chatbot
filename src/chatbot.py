class Chatbot:
    def __init__(self, model):
        self.model = model  # This will be an instance of the AI model you're using

    def get_response(self, query):
        # Process the query and interact with the AI model to get a response
        response = self.model.generate_answer(query)
        return response
