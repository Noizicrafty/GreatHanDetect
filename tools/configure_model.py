"""Create a model-bound application config from static ONNX and an optional compiled program."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import onnx
import yaml
from greathan.config import load_config,public_config,save_config,ROOT


def configure(base,onnx_file,out,role='detector',classes=None,program=None,remote_program=None,scenario=None):
    config=public_config(load_config(base))
    graph=onnx.load(str(onnx_file))
    initialized={x.name for x in graph.graph.initializer}
    inputs=[x for x in graph.graph.input if x.name not in initialized]
    if len(inputs)!=1 or len(graph.graph.output)!=1:raise ValueError('Application decoder requires one image input and one assembled output')
    def spec(value):return {'shape':[d.dim_value for d in value.type.tensor_type.shape.dim],'dtype':'float32'}
    input_spec=spec(inputs[0]);size=input_spec['shape'][-1]
    if input_spec['shape']!=[1,3,size,size]:raise ValueError('Expected static square RGB NCHW image input')
    model={**config['models'].get(role,{}),'onnx':str(Path(onnx_file).resolve()),'input_size':size,
           'input_scale':1,'inputs':{inputs[0].name:input_spec},'outputs':{graph.graph.output[0].name:spec(graph.graph.output[0])},
           'onnx_sha256':hashlib.sha256(Path(onnx_file).read_bytes()).hexdigest()}
    if classes is not None:
        model['class_names']=classes;model['decoder']='yolo_objects'
        if spec(graph.graph.output[0])['shape'][1]!=4+len(classes):raise ValueError('Output channel count differs from ordered classes')
    if program:
        model['local_program']=str(Path(program).resolve());model['program_sha256']=hashlib.sha256(Path(program).read_bytes()).hexdigest()
    else:
        model.pop('program_sha256',None)
        model.pop('local_program',None);model.pop('remote_program',None)
        config['backend']='onnx_cpu'
    if remote_program:
        model['remote_program']=remote_program
        if program:config['backend']='linq_amqp'
    config['models'][role]=model
    if scenario:config['scenario']=scenario
    save_config(out,config)
    return config


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='configs/application.yaml');parser.add_argument('--onnx',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--role',default='detector')
    parser.add_argument('--classes',nargs='+');parser.add_argument('--program');parser.add_argument('--remote-program')
    parser.add_argument('--scenario',choices=['danger_zone','smoking'])
    args=parser.parse_args()
    configure(args.base,args.onnx,args.out,args.role,args.classes,args.program,args.remote_program,args.scenario)
    print(args.out)
