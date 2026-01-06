import json
import logging
from datetime import datetime
from pathlib import Path
from signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('demo_executor')

def run_demo_session(balance=100.0, lot_sizes=[0.05, 0.10]):
    logger.info("=" * 60)
    logger.info(f"🚀 STARTING DEMO SESSION - ACCOUNT BALANCE: ${balance}")
    logger.info("=" * 60)
    
    # Initialize Generator
    generator = SignalGenerator()
    signals = generator.generate_all_signals(account_value=balance)
    
    demo_trades = []
    
    for lot_size in lot_sizes:
        logger.info(f"\n--- SIMULATING LOT SIZE: {lot_size} ---")
        
        for signal in signals:
            if signal['signal'] == 'HOLD':
                continue
                
            ticker = signal['ticker']
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp = signal['take_profit']
            
            # Unit calculation for Forex
            # Standard Lot = 100,000 units. 0.10 = 10,000. 0.05 = 5,000.
            units = lot_size * 100000
            
            # Special handling for Gold (GC=F) units
            # Usually 1 lot = 100oz. 0.10 lot = 10oz.
            if ticker == 'GC=F':
                units = lot_size * 100
            
            # Calculate Risk in Dollars
            risk_per_unit = abs(entry - sl)
            total_risk_dollars = risk_per_unit * units
            risk_pct = (total_risk_dollars / balance) * 100
            
            # Calculate Potential Profit
            profit_per_unit = abs(tp - entry)
            total_profit_dollars = profit_per_unit * units
            
            trade = {
                'ticker': ticker,
                'lot_size': lot_size,
                'units': units,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'risk_dollars': round(total_risk_dollars, 2),
                'potential_profit': round(total_profit_dollars, 2),
                'risk_pct': round(risk_pct, 2),
                'status': 'OPEN'
            }
            
            demo_trades.append(trade)
            
            logger.info(f"TRADE: {ticker} {signal['signal']} {lot_size} lots")
            logger.info(f"   Entry: {entry} | SL: {sl} | TP: {tp}")
            logger.info(f"   💰 Risk: ${trade['risk_dollars']} ({trade['risk_pct']}%)")
            logger.info(f"   🎯 Potential Profit: ${trade['potential_profit']}")
            
            if risk_pct > 50:
                logger.warning(f"   ⚠️ EXTREME RISK: This trade risks {trade['risk_pct']}% of your account!")
            elif risk_pct > 100:
                logger.error(f"   🛑 INSUFFICIENT FUNDS: Margin call likely before SL hit.")

    # Save demo results
    output_path = Path('signals/demo_portfolio.json')
    with open(output_path, 'w') as f:
        json.dump({
            'session_start': datetime.now().isoformat(),
            'initial_balance': balance,
            'trades': demo_trades
        }, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Demo Session Executed. Details saved to {output_path}")
    logger.info("=" * 60)

if __name__ == '__main__':
    run_demo_session()
