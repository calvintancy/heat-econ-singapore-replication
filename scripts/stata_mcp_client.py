"""Run local Stata through the real MCP stdio protocol with durable receipts.

Execute in an environment containing mcp>=1.9,<2. The server interpreter is an
explicit CLI argument; no global MCP configuration or profile is changed.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stata_key_results/mcp'


async def run(args):
    receipt_dir=Path(args.receipt_dir).resolve()
    receipt_dir.mkdir(parents=True,exist_ok=True)
    stem=args.receipt
    record={'started_utc':datetime.now(timezone.utc).isoformat(),'transport':'MCP stdio',
        'client_mcp_version':importlib.metadata.version('mcp'),'server_command':args.server_python,
        'server_args':['-m','mcp_stata'],'stata_path':args.stata_path,'calls':[],'status':'running'}
    path=receipt_dir/f'{stem}.json'
    def save():path.write_text(json.dumps(record,indent=2,ensure_ascii=False,default=str)+'\n')
    save()
    env=os.environ.copy();env.update(STATA_PATH=args.stata_path,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
    server=StdioServerParameters(command=args.server_python,args=['-m','mcp_stata'],env=env,cwd=str(ROOT))
    try:
        with (receipt_dir/f'{stem}_server_stderr.log').open('w') as errors:
            async with stdio_client(server,errlog=errors) as streams:
                async with ClientSession(*streams,read_timeout_seconds=timedelta(seconds=args.timeout)) as session:
                    init=await session.initialize();record['initialization']=init.model_dump(mode='json');save()
                    inventory=await session.list_tools();record['tools']=inventory.model_dump(mode='json');save()
                    print('MCP initialized; tools: '+', '.join(item.name for item in inventory.tools),flush=True)
                    if args.detect:
                        result=await session.call_tool('stata_manage_session',{'action':'detect','include_packages':True,'as_json':True})
                        record['calls'].append({'tool':'stata_manage_session','arguments':{'action':'detect','include_packages':True,'as_json':True},'response':result.model_dump(mode='json')});save()
                        print(json.dumps(record['calls'][-1],ensure_ascii=False),flush=True)
                    if args.code or args.do_file:
                        request={'code':str(Path(args.do_file).resolve()) if args.do_file else args.code,
                            'is_file':bool(args.do_file),'cwd':str(ROOT),'session_id':'default','as_json':True,
                            'echo':True,'max_output_lines':80,'allow_unsafe_paths':True}
                        call={'tool':'stata_run','arguments':request,'status':'running'};record['calls'].append(call);save()
                        result=await session.call_tool('stata_run',request)
                        call.update(status='returned',response=result.model_dump(mode='json'));save()
                        print(json.dumps(call,ensure_ascii=False),flush=True)
                        payload=result.structuredContent or {}
                        if 'result' in payload and isinstance(payload['result'],str):payload=json.loads(payload['result'])
                        if result.isError or payload.get('success') is False or payload.get('data',{}).get('rc',0)!=0:
                            raise RuntimeError('Stata MCP command failed; inspect the retained protocol response and native log.')
                    record['status']='completed';save()
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save();raise
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-python',default=sys.executable)
    parser.add_argument('--stata-path',default='/Applications/Stata/StataBE.app/Contents/MacOS/StataBE')
    parser.add_argument('--receipt',default='probe')
    parser.add_argument('--receipt-dir',default=str(OUT))
    parser.add_argument('--timeout',type=int,default=180)
    parser.add_argument('--detect',action='store_true')
    group=parser.add_mutually_exclusive_group();group.add_argument('--code');group.add_argument('--do-file')
    asyncio.run(run(parser.parse_args()))
