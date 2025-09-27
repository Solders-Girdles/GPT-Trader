#!/usr/bin/env python3
"""
Web Dashboard Server for Paper Trading & Bot Management
Serves the dashboard and provides API endpoints for session data and BotManager.
"""

import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import threading
import time

# Load environment (CDP keys) so market data works for paper execution
env_file = Path(__file__).parent.parent / '.env.production'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"')
                if key == 'COINBASE_CDP_PRIVATE_KEY':
                    private_key_lines = [value] if value else []
                    for next_line in f:
                        next_line = next_line.strip()
                        private_key_lines.append(next_line)
                        if 'END EC PRIVATE KEY' in next_line:
                            break
                    value = '\n'.join(private_key_lines)
                os.environ[key] = value

# Bot manager imports
try:
    # Ensure project root is on path
    sys.path.insert(0, str(Path(__file__).parent.parent))
except Exception:
    pass

from bot_v2.orchestration.bot_manager import BotManager, BotConfig, RiskConfig
from bot_v2.orchestration.session_store import SessionStore
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.config_store import ConfigStore
from dataclasses import asdict

# Global manager + store
MANAGER = BotManager()
STORE = SessionStore()
EVENTS = EventStore()
CONFIGS = ConfigStore()

# Load persisted bots and optionally auto-start
def _load_persisted_bots():
    bots = CONFIGS.load_bots()
    for b in bots:
        try:
            risk = b.get('risk') or {}
            cfg = BotConfig(
                bot_id=b['bot_id'],
                name=b.get('name') or b['bot_id'],
                symbols=b.get('symbols') or [],
                strategy=b.get('strategy', 'scalp'),
                strategy_params=b.get('strategy_params') or {},
                capital=float(b.get('capital', 10000)),
                risk=RiskConfig(
                    max_positions=int(risk.get('max_positions', 6)),
                    max_position_size=float(risk.get('max_position_size', 0.2)),
                    stop_loss=float(risk.get('stop_loss', 0.04)),
                    take_profit=float(risk.get('take_profit', 0.08)),
                    commission=float(risk.get('commission', 0.004)),
                    slippage=float(risk.get('slippage', 0.001)),
                ),
                loop_sleep=float(b.get('loop_sleep', 5.0)),
                auto_start=bool(b.get('auto_start', False)),
            )
            MANAGER.add_bot(cfg)
            if cfg.auto_start:
                MANAGER.start(cfg.bot_id)
        except Exception:
            continue

_load_persisted_bots()


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler for dashboard and API endpoints."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            # Serve dashboard
            self.serve_dashboard()
        elif parsed_path.path == '/api/sessions':
            # List available session files
            self.serve_session_list()
        elif parsed_path.path == '/api/session':
            # Serve latest session data
            self.serve_session_data(parsed_path.query)
        elif parsed_path.path == '/api/bots':
            self.serve_bots()
        elif parsed_path.path.startswith('/api/bots/'):
            if parsed_path.path.endswith('/events'):
                self.serve_bot_events(parsed_path.path, parsed_path)
            else:
                self.serve_bot_detail(parsed_path.path)
        else:
            # Default file serving
            super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)
        try:
            content_len = int(self.headers.get('Content-Length') or 0)
        except Exception:
            content_len = 0
        body = b''
        if content_len > 0:
            body = self.rfile.read(content_len)
        payload = {}
        if body:
            try:
                payload = json.loads(body.decode('utf-8'))
            except Exception:
                payload = {}

        if parsed_path.path == '/api/bots':
            self.create_bot(payload)
        elif parsed_path.path.startswith('/api/bots/') and self.command == 'DELETE':
            bot_id = parsed_path.path.split('/')[-1]
            self.delete_bot(bot_id)
        elif parsed_path.path.endswith('/start'):
            bot_id = parsed_path.path.split('/')[-2]
            MANAGER.start(bot_id)
            self.send_json({'status': 'ok', 'bot_id': bot_id})
        elif parsed_path.path.endswith('/stop'):
            bot_id = parsed_path.path.split('/')[-2]
            MANAGER.stop(bot_id)
            self.send_json({'status': 'ok', 'bot_id': bot_id})
        elif parsed_path.path.endswith('/set_auto_start'):
            bot_id = parsed_path.path.split('/')[-2]
            enabled = bool(payload.get('enabled', True))
            # update store
            CONFIGS.update_bot(bot_id, {'auto_start': enabled})
            self.send_json({'status': 'ok', 'bot_id': bot_id, 'auto_start': enabled})
        elif parsed_path.path.endswith('/snapshot'):
            bot_id = parsed_path.path.split('/')[-2]
            bot = MANAGER.get_bot(bot_id)
            path = STORE.save_bot_snapshot(bot)
            self.send_json({'status': 'ok', 'bot_id': bot_id, 'file': str(path)})
        else:
            self.send_error(404, 'Unsupported POST path')

    def do_PATCH(self):
        parsed_path = urlparse(self.path)
        try:
            content_len = int(self.headers.get('Content-Length') or 0)
        except Exception:
            content_len = 0
        body = b''
        if content_len > 0:
            body = self.rfile.read(content_len)
        payload = {}
        if body:
            try:
                payload = json.loads(body.decode('utf-8'))
            except Exception:
                payload = {}

        if parsed_path.path.startswith('/api/bots/'):
            bot_id = parsed_path.path.split('/')[-1]
            self.update_bot(bot_id, payload)
        else:
            self.send_error(404, 'Unsupported PATCH path')
    
    def serve_dashboard(self):
        """Serve the dashboard HTML."""
        dashboard_path = Path(__file__).parent / 'dashboard.html'
        
        if dashboard_path.exists():
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open(dashboard_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, 'Dashboard not found')
    
    def serve_session_data(self, query: str = ""):
        """Serve the latest session data as JSON."""
        results_dir = Path(__file__).parent.parent / 'results'
        params = {}
        if query:
            try:
                for kv in query.split('&'):
                    if not kv:
                        continue
                    k, v = kv.split('=', 1)
                    params[k] = v
            except Exception:
                params = {}
        
        # Find the most recent session file
        session_files = list(results_dir.glob('live_*.json')) + list(results_dir.glob('*.json'))
        
        if not session_files:
            self.send_response(200)
            self.send_json({'error': 'No session data found'})
            return
        
        # Pick by name if provided, otherwise most recent
        file_param = params.get('file')
        if file_param:
            candidate = results_dir / file_param
            latest_file = candidate if candidate.exists() else max(session_files, key=lambda f: f.stat().st_mtime)
        else:
            latest_file = max(session_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            self.send_json(data)
        except Exception as e:
            self.send_error(500, f'Error loading session data: {e}')

    def serve_session_list(self):
        """List available sessions with metadata."""
        results_dir = Path(__file__).parent.parent / 'results'
        files = []
        for f in sorted(results_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                files.append({
                    'name': f.name,
                    'mtime': f.stat().st_mtime,
                    'size': f.stat().st_size
                })
            except Exception:
                continue
        self.send_json({'sessions': files})

    # BotManager API
    def serve_bots(self):
        bots = []
        for b in MANAGER.list_bots():
            bots.append({
                'id': b.config.bot_id,
                'name': b.config.name,
                'status': b.status,
                'auto_start': b.config.auto_start,
                'config': {
                    'symbols': b.config.symbols,
                    'strategy': b.config.strategy,
                    'strategy_params': b.config.strategy_params,
                    'capital': b.config.capital,
                    'risk': {
                        'max_positions': b.config.risk.max_positions,
                        'max_position_size': b.config.risk.max_position_size,
                        'stop_loss': b.config.risk.stop_loss,
                        'take_profit': b.config.risk.take_profit,
                        'commission': b.config.risk.commission,
                        'slippage': b.config.risk.slippage,
                    },
                    'loop_sleep': b.config.loop_sleep,
                },
                'metrics': {
                    'trades': b.metrics.trades,
                    'signals': b.metrics.signals,
                    'buy_signals': b.metrics.buy_signals,
                    'sell_signals': b.metrics.sell_signals,
                    'hold_signals': b.metrics.hold_signals,
                    'execution_rate': b.metrics.execution_rate,
                    'trades_per_hour': b.metrics.trades_per_hour,
                    'signals_per_hour': b.metrics.signals_per_hour,
                    'start_time': b.metrics.start_time.isoformat(),
                    'last_update': b.metrics.last_update.isoformat() if b.metrics.last_update else None,
                }
            })
        self.send_json({'bots': bots})

    def serve_bot_detail(self, path: str):
        parts = path.split('/')
        bot_id = parts[-1]
        try:
            b = MANAGER.get_bot(bot_id)
        except Exception:
            self.send_error(404, f'bot {bot_id} not found')
            return
        payload = {
            'id': b.config.bot_id,
            'name': b.config.name,
            'status': b.status,
            'config': {
                'symbols': b.config.symbols,
                'strategy': b.config.strategy,
                'strategy_params': b.config.strategy_params,
                'capital': b.config.capital,
                'risk': {
                    'max_positions': b.config.risk.max_positions,
                    'max_position_size': b.config.risk.max_position_size,
                    'stop_loss': b.config.risk.stop_loss,
                    'take_profit': b.config.risk.take_profit,
                    'commission': b.config.risk.commission,
                    'slippage': b.config.risk.slippage,
                },
                'loop_sleep': b.config.loop_sleep,
            },
            'metrics': {
                'trades': b.metrics.trades,
                'signals': b.metrics.signals,
                'buy_signals': b.metrics.buy_signals,
                'sell_signals': b.metrics.sell_signals,
                'hold_signals': b.metrics.hold_signals,
                'execution_rate': b.metrics.execution_rate,
                'trades_per_hour': b.metrics.trades_per_hour,
                'signals_per_hour': b.metrics.signals_per_hour,
                'start_time': b.metrics.start_time.isoformat(),
                'last_update': b.metrics.last_update.isoformat() if b.metrics.last_update else None,
            }
        }
        self.send_json(payload)

    def serve_bot_events(self, path: str, parsed):
        parts = path.split('/')
        bot_id = parts[-2]
        # parse limit from query if present
        limit = 50
        types = None
        try:
            qs = parsed.query or ''
            if 'limit=' in qs:
                for kv in qs.split('&'):
                    if kv.startswith('limit='):
                        limit = int(kv.split('=', 1)[1])
                        break
            if 'types=' in qs:
                for kv in qs.split('&'):
                    if kv.startswith('types='):
                        raw = kv.split('=', 1)[1]
                        types = [t.strip() for t in raw.split(',') if t.strip()]
                        break
        except Exception:
            pass
        events = EVENTS.tail(bot_id, limit=limit, types=types)
        self.send_json({'events': events})

    def create_bot(self, payload: dict):
        try:
            symbols = payload.get('symbols') or []
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(',') if s.strip()]
            risk = payload.get('risk') or {}
            cfg = BotConfig(
                bot_id=payload['id'],
                name=payload.get('name') or payload['id'],
                symbols=symbols,
                strategy=payload.get('strategy', 'scalp'),
                strategy_params=payload.get('strategy_params') or {},
                capital=float(payload.get('capital', 10000)),
                risk=RiskConfig(
                    max_positions=int(risk.get('max_positions', 6)),
                    max_position_size=float(risk.get('max_position_size', 0.2)),
                    stop_loss=float(risk.get('stop_loss', 0.04)),
                    take_profit=float(risk.get('take_profit', 0.08)),
                    commission=float(risk.get('commission', 0.004)),
                    slippage=float(risk.get('slippage', 0.001)),
                ),
                loop_sleep=float(payload.get('loop_sleep', 5.0)),
                auto_start=bool(payload.get('auto_start', False)),
            )
            # Validate config
            errors = self._validate_config(asdict(cfg))
            if errors:
                self.send_json({'status': 'error', 'errors': errors}, code=400)
                return
            MANAGER.add_bot(cfg)
            CONFIGS.add_bot(asdict(cfg))
            self.send_json({'status': 'ok', 'id': cfg.bot_id})
        except Exception as e:
            self.send_error(400, f'create bot failed: {e}')

    def delete_bot(self, bot_id: str):
        try:
            MANAGER.stop(bot_id)
        except Exception:
            pass
        CONFIGS.remove_bot(bot_id)
        self.send_json({'status': 'ok', 'id': bot_id})

    def update_bot(self, bot_id: str, payload: dict):
        try:
            b = MANAGER.get_bot(bot_id)
        except Exception:
            self.send_error(404, f'bot {bot_id} not found')
            return
        # Track running state
        was_running = (b.status == 'running')
        try:
            if was_running:
                MANAGER.stop(bot_id)
        except Exception:
            pass

        # Apply updates to config
        if 'name' in payload:
            b.config.name = str(payload['name'])
        if 'loop_sleep' in payload:
            try:
                b.config.loop_sleep = float(payload['loop_sleep'])
            except Exception:
                pass
        if 'auto_start' in payload:
            b.config.auto_start = bool(payload['auto_start'])
        # Risk updates
        risk = payload.get('risk') or {}
        for k in ('max_positions','max_position_size','stop_loss','take_profit','commission','slippage'):
            if k in risk:
                try:
                    setattr(b.config.risk, k, type(getattr(b.config.risk, k))(risk[k]))
                except Exception:
                    pass
        # Strategy params (partial merge)
        if 'strategy_params' in payload and isinstance(payload['strategy_params'], dict):
            b.config.strategy_params.update(payload['strategy_params'])

        # Validate new config before persisting
        new_cfg = asdict(b.config)
        errors = self._validate_config(new_cfg)
        if errors:
            # revert running state if we stopped
            if was_running:
                try:
                    MANAGER.start(bot_id)
                except Exception:
                    pass
            self.send_json({'status': 'error', 'errors': errors}, code=400)
            return
        # Persist
        CONFIGS.update_bot(bot_id, new_cfg)

        # Restart based on previous state or auto_start
        if was_running or b.config.auto_start:
            try:
                MANAGER.start(bot_id)
            except Exception:
                pass
        self.send_json({'status':'ok','id':bot_id})

    # Basic server-side validation for pragmatic guardrails
    def _validate_config(self, cfg: dict) -> list:
        errs = []
        symbols = cfg.get('symbols') or []
        if not isinstance(symbols, list):
            errs.append('symbols must be a list')
        if isinstance(symbols, list) and len(symbols) == 0:
            errs.append('at least one symbol is required')
        loop_sleep = cfg.get('loop_sleep', 5.0)
        try:
            ls = float(loop_sleep)
            if ls <= 0 or ls > 120:
                errs.append('loop_sleep must be between 0 and 120 seconds')
        except Exception:
            errs.append('loop_sleep must be a number')
        risk = cfg.get('risk') or {}
        try:
            mp = int(risk.get('max_positions', 1))
            if mp < 1 or mp > 50:
                errs.append('risk.max_positions must be 1..50')
        except Exception:
            errs.append('risk.max_positions must be integer')
        try:
            mps = float(risk.get('max_position_size', 0.2))
            if not (0 < mps <= 1):
                errs.append('risk.max_position_size must be in (0,1]')
        except Exception:
            errs.append('risk.max_position_size must be number')
        for fld in ('stop_loss','take_profit','commission','slippage'):
            try:
                v = float(risk.get(fld, 0.0))
                if v < 0 or v >= 1:
                    errs.append(f'risk.{fld} must be in [0,1)')
            except Exception:
                errs.append(f'risk.{fld} must be number')
        # SL/TP sanity: not both zero; TP > SL suggested
        try:
            sl = float(risk.get('stop_loss', 0.0))
            tp = float(risk.get('take_profit', 0.0))
            if sl == 0 and tp == 0:
                errs.append('risk.stop_loss or risk.take_profit should be > 0')
        except Exception:
            pass
        return errs

    # Utility
    def send_json(self, payload: dict, code: int = 200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(port: int = 8888):
    """Run the dashboard server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print("=" * 70)
    print("PAPER TRADING DASHBOARD SERVER")
    print("=" * 70)
    print(f"Dashboard URL: http://localhost:{port}")
    print("API Endpoint: http://localhost:{port}/api/session")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trading Dashboard Server")
    parser.add_argument('--port', type=int, default=8888,
                      help='Port to run server on (default: 8888)')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    run_server(args.port)


if __name__ == "__main__":
    main()
