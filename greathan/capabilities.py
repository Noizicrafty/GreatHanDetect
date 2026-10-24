"""Explicit capability catalogue. Demonstration events never claim model execution."""
import uuid

EXTRA_SCENARIOS={
    'helmet':{'name':'Отсутствие каски','event_type':'demo_helmet'},
    'vest':{'name':'Отсутствие сигнального жилета','event_type':'demo_vest'},
    'fall':{'name':'Падение человека','event_type':'demo_fall'},
}


def catalogue():
    primary=[{'id':'danger_zone','name':'Вход в опасную зону','state':'implemented','mode':'neural_and_rule'},
             {'id':'smoking','name':'Предполагаемое курение','state':'implemented','mode':'neural_and_rule'}]
    return primary+[
        {'id':key,**value,'state':'model_not_connected','mode':'demo_only','trigger':'smoking',
         'provider_reference':'https://github.com/PaddlePaddle/PaddleDetection',
         'note':'Catalogue / demonstration only. No PPE or fall model is executed.'}
        for key,value in EXTRA_SCENARIOS.items()]


def demo_events(parent,config):
    settings=config.get('demo_additional',{})
    if not settings.get('enabled') or parent['event_type']!='smoking':return []
    results=[]
    for scenario in settings.get('scenarios',[]):
        spec=EXTRA_SCENARIOS[scenario]
        event={**parent,'id':uuid.uuid4().hex,'episode_id':parent['episode_id']+':demo:'+scenario,
               'event_type':spec['event_type'],'backend_kind':'demo_simulated',
               'model_version':'NO_MODEL_EXECUTED_DEMONSTRATION',
               'confidence':0.0,'reason':'ДЕМО: иллюстрация дополнительного сценария по событию курения; отдельная модель не исполнялась.'}
        results.append(event)
    return results
