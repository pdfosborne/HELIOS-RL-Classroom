from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor


# StateAdapter includes static methods for adapters
from helios.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder



class ClassroomALanguage:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self):
        self.student_features = pd.read_csv('./adapters/language_data/student_features.csv')
        self.encoder = LanguageEncoder()
    
    def adapter(self, state_x:int, state_y:int, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
       
        features = self.student_features[(self.student_features['state_x']==state_x)&(self.student_features['state_y']==state_y)].iloc[:,2:]
        student_features = features.values[0]
        
        stud_type = student_features[0]
        stud_hair_colour = student_features[1]
        stud_hair_style = student_features[2]
        stud_upperclothing_type = student_features[3]
        stud_upperclothing_colour = student_features[4]
        stud_lowerclothing_type = student_features[5]
        stud_lowerclothing_colour = student_features[6]
        stud_piercings = student_features[7]
        stud_gender = student_features[8]
        


        # Covert numeric dict to a list of strings describing player positions
        state:List[str] = []
        if stud_type == 'trash':
            full_str = 'This is a trash can.'
        elif stud_type == 'recycling':
            full_str = 'This is a recycling bin.'
        elif stud_type == 'teacher':
            full_str = ('A ' + stud_gender + ' teacher that has ' + stud_hair_style + ' ' + stud_hair_colour + ' hair and is wearing a ' + 
                        stud_upperclothing_colour + ' ' + stud_upperclothing_type + ', ' + stud_lowerclothing_colour + ' ' + stud_lowerclothing_type + 
                        ' and has ' + stud_piercings + ' piercings.')
        else:
            full_str = ('A ' + stud_gender + ' student that has ' + stud_hair_style + ' ' + stud_hair_colour + ' hair and is wearing a ' + 
                        stud_upperclothing_colour + ' ' + stud_upperclothing_type + ', ' + stud_lowerclothing_colour + ' ' + stud_lowerclothing_type + 
                        ' and has ' + stud_piercings + ' piercings.')
        state.append(full_str) 
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in ClassroomALanguage._cached_state_idx):
                    ClassroomALanguage._cached_state_idx[sent] = len(ClassroomALanguage._cached_state_idx)
                state_indexed.append(ClassroomALanguage._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded
    
    # def sample():
        # board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        # legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
        #                'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
        #                'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
        #                'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        # episode_action_history = ['e2e4', 'c7c5']
        # adapter = BoardToLanguageAdapter()
        # state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        # state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)
        # return state,state_encoded