import hashlib
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import warnings

warnings.filterwarnings('ignore')


# ==================== 第一部分：基础银行系统（原功能保留但优化） ====================

class PasswordMixin:
    """密码处理混入类"""

    def is_valid(self, pw):
        while len(pw) != 6 or not pw.isdigit():
            pw = input('密码需为6位数字，请重新输入：')
        return pw

    def to_md5(self, pw):
        md5_hash = hashlib.md5()
        md5_hash.update(pw.encode('utf-8'))
        return md5_hash.hexdigest()


# 扩展的数据库结构：存储完整客户信息
database = {}
transactions = []  # 新增：存储所有交易记录


class UserManager(PasswordMixin):
    """用户管理类 - 扩展以支持更多客户特征"""
    id_counter = 88888888

    def __init__(self):
        self.current_id = None
        self.current_name = None

    def create_account(self):
        """创建账户 - 扩展为收集更多客户特征"""
        print("\n=== 创建新账户 ===")

        # 基础信息
        self.name = input('请输入姓名：')
        pw = input('请输入密码：')
        while not pw.isdigit():
            pw = input('密码需为6位数字，请重新输入：')

        initial_deposit = float(input('请输入初始存款金额：'))
        while initial_deposit < 0:
            initial_deposit = float(input('金额不合法，请重新输入：'))

        # 随机生成信用评估相关特征（模拟场景下）
        age = random.randint(22, 60)
        income_level = random.choice([1, 2, 3, 4, 5])  # 收入等级 1-5
        job_stability = random.randint(1, 10)  # 工作稳定性 1-10
        credit_history = random.randint(0, 5)  # 信用历史长度（年）

        has_previous_loan = random.choice([0, 1])  # 是否有历史贷款
        existing_debt = random.uniform(0, 50000) if has_previous_loan else 0

        # 存储扩展客户信息
        client_id = UserManager.id_counter
        database[client_id] = {
            'basic_info': {
                'name': self.name,
                'password': self.to_md5(self.is_valid(pw)),
                'balance': initial_deposit,
                'created_date': datetime.now().strftime('%Y-%m-%d')
            },
            'credit_features': {
                'age': age,
                'income_level': income_level,
                'job_stability': job_stability,
                'credit_history': credit_history,
                'has_previous_loan': has_previous_loan,
                'existing_debt': existing_debt,
                'initial_deposit': initial_deposit
            },
            'behavior_metrics': {
                'deposit_count': 0,
                'withdrawal_count': 0,
                'transfer_count': 0,
                'avg_transaction_amount': 0,
                'max_balance': initial_deposit,
                'min_balance': initial_deposit
            }
        }

        UserManager.id_counter += 1
        print(f'\n✅ 账户创建成功！')
        print(f'   账户号：{client_id}')
        print(f'   姓名：{self.name}')
        print(f'   初始余额：¥{initial_deposit:.2f}')
        print(f'   生成信用特征：年龄{age}岁，收入等级{income_level}，工作稳定性{job_stability}/10')
        return client_id


# ==================== 第二部分：交易系统（增强以记录行为数据） ====================

class Account(UserManager):
    """账户操作类 - 扩展以记录交易行为"""

    def log_transaction(self, client_id, transaction_type, amount, target_id=None):
        """记录交易日志"""
        transaction = {
            'timestamp': datetime.now(),
            'client_id': client_id,
            'type': transaction_type,
            'amount': amount,
            'balance_after': database[client_id]['basic_info']['balance'],
            'target_id': target_id
        }
        transactions.append(transaction)

        # 更新行为指标
        if transaction_type == 'deposit':
            database[client_id]['behavior_metrics']['deposit_count'] += 1
        elif transaction_type == 'withdrawal':
            database[client_id]['behavior_metrics']['withdrawal_count'] += 1
        elif transaction_type == 'transfer':
            database[client_id]['behavior_metrics']['transfer_count'] += 1

        # 更新余额统计
        current_balance = database[client_id]['basic_info']['balance']
        database[client_id]['behavior_metrics']['max_balance'] = max(
            database[client_id]['behavior_metrics']['max_balance'],
            current_balance
        )
        database[client_id]['behavior_metrics']['min_balance'] = min(
            database[client_id]['behavior_metrics']['min_balance'],
            current_balance
        )

    def deposit(self):
        """存款"""
        client_id = int(input('请输入卡号：'))
        if client_id not in database:
            print('❌ 卡号不存在')
            return

        # 验证密码
        pw = self.to_md5(self.is_valid(input('请输入密码：')))
        if pw != database[client_id]['basic_info']['password']:
            print('❌ 密码错误')
            return

        amount = float(input('请输入存款金额：'))
        if amount <= 0:
            print('❌ 金额必须大于0')
            return

        database[client_id]['basic_info']['balance'] += amount
        self.log_transaction(client_id, 'deposit', amount)
        print(f'✅ 成功存入¥{amount:.2f}，当前余额：¥{database[client_id]["basic_info"]["balance"]:.2f}')

    def withdraw(self):
        """取款"""
        client_id = int(input('请输入卡号：'))
        if client_id not in database:
            print('❌ 卡号不存在')
            return

        # 验证密码
        pw = self.to_md5(self.is_valid(input('请输入密码：')))
        if pw != database[client_id]['basic_info']['password']:
            print('❌ 密码错误')
            return

        amount = float(input('请输入取款金额：'))
        balance = database[client_id]['basic_info']['balance']

        if amount <= 0:
            print('❌ 金额必须大于0')
            return
        if amount > balance:
            print(f'❌ 余额不足，当前余额：¥{balance:.2f}')
            return

        database[client_id]['basic_info']['balance'] -= amount
        self.log_transaction(client_id, 'withdrawal', amount)
        print(f'✅ 成功取出¥{amount:.2f}，当前余额：¥{database[client_id]["basic_info"]["balance"]:.2f}')


# ==================== 第三部分：信用评分系统（新增核心） ====================

class CreditScoringSystem:
    """信用评分与风险预测系统"""

    def __init__(self):
        self.model = None
        self.feature_columns = None

    def simulate_client_behavior(self, num_clients=100, days=90):
        """
        模拟客户行为以生成训练数据
        返回：(特征DataFrame, 违约标签)
        """
        print(f"\n🔧 开始模拟{num_clients}个客户{days}天的行为...")

        features = []
        labels = []

        for client_id in range(1, num_clients + 1):
            # 生成客户特征
            age = random.randint(22, 65)
            income = random.randint(20000, 150000)
            job_stability = random.randint(1, 10)
            credit_history = random.randint(0, 20)
            existing_debt_ratio = random.random() * 0.8  # 现有债务占收入比例

            # 初始金融行为
            initial_balance = random.randint(1000, 50000)
            balance = initial_balance

            # 行为指标
            deposit_count = 0
            withdrawal_count = 0
            transaction_amounts = []
            low_balance_days = 0

            # 模拟days天的行为
            for day in range(days):
                # 模拟每日交易
                if random.random() < 0.3:  # 30%概率有交易
                    transaction_type = random.choice(['deposit', 'withdrawal'])
                    amount = random.randint(100, 5000)

                    if transaction_type == 'deposit':
                        balance += amount
                        deposit_count += 1
                    else:  # withdrawal
                        if amount <= balance:
                            balance -= amount
                            withdrawal_count += 1

                    transaction_amounts.append(amount)

                # 检查低余额
                if balance < 1000:
                    low_balance_days += 1

            # 计算衍生特征
            avg_transaction = np.mean(transaction_amounts) if transaction_amounts else 0
            balance_volatility = np.std([initial_balance, balance]) if days > 1 else 0

            # 组合特征
            client_features = [
                age,
                income,
                job_stability,
                credit_history,
                existing_debt_ratio,
                initial_balance,
                deposit_count,
                withdrawal_count,
                avg_transaction,
                balance_volatility,
                low_balance_days / days if days > 0 else 0,
                balance  # 最终余额
            ]

            # 生成标签（违约概率与特征相关）
            # 违约风险因素：高债务比、低收入、低余额、频繁取款
            risk_score = (
                    existing_debt_ratio * 0.3 +
                    (1 - income / 150000) * 0.2 +
                    (balance < 2000) * 0.3 +
                    (withdrawal_count > deposit_count * 1.5) * 0.2
            )

            # 违约标签（风险评分>0.5的客户更可能违约）
            will_default = 1 if risk_score > 0.5 else 0

            features.append(client_features)
            labels.append(will_default)

        # 特征列名
        self.feature_columns = [
            'age', 'income', 'job_stability', 'credit_history', 'existing_debt_ratio',
            'initial_balance', 'deposit_count', 'withdrawal_count', 'avg_transaction',
            'balance_volatility', 'low_balance_ratio', 'final_balance'
        ]

        print(f"✅ 模拟完成，生成{len(features)}条样本，违约率：{sum(labels) / len(labels):.1%}")
        return pd.DataFrame(features, columns=self.feature_columns), np.array(labels)

    def train_model(self, X, y, test_size=0.2):
        """训练信用评分模型"""
        print("\n🤖 开始训练信用评分模型...")

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_train, y_train)

        # 评估模型
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n📊 模型评估结果：")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🔍 特征重要性排名：")
        print(feature_importance.head(10).to_string(index=False))

        return self.model

    def predict_credit_risk(self, client_features):
        """预测单个客户的信用风险"""
        if self.model is None:
            print("❌ 请先训练模型")
            return None

        # 确保特征顺序正确
        if isinstance(client_features, dict):
            client_features = [client_features[col] for col in self.feature_columns]

        features_array = np.array(client_features).reshape(1, -1)
        probability = self.model.predict_proba(features_array)[0, 1]

        risk_level = "低风险" if probability < 0.3 else "中风险" if probability < 0.7 else "高风险"

        return {
            'default_probability': probability,
            'risk_level': risk_level,
            'credit_score': int((1 - probability) * 850),  # 转换为信用分（300-850范围）
            'recommendation': '建议批准' if probability < 0.5 else '建议谨慎' if probability < 0.7 else '建议拒绝'
        }

    def save_model(self, filename='credit_scoring_model.pkl'):
        """保存模型"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'features': self.feature_columns
            }, f)
        print(f"💾 模型已保存到 {filename}")


# ==================== 第四部分：主界面与集成 ====================

def main():
    """主程序"""
    credit_system = CreditScoringSystem()

    print("=" * 60)
    print("       智能银行系统 - 信用评分模拟平台")
    print("=" * 60)

    while True:
        print("\n" + "=" * 40)
        print("          主菜单")
        print("=" * 40)
        print("1. 创建账户（扩展特征）")
        print("2. 存款")
        print("3. 取款")
        print("4. 模拟客户行为数据")
        print("5. 训练信用评分模型")
        print("6. 预测客户信用风险")
        print("7. 查看系统统计")
        print("8. 保存模型")
        print("9. 退出")
        print("-" * 40)

        choice = input("请选择操作 (1-9): ").strip()

        if choice == '1':
            u = UserManager()
            u.create_account()

        elif choice == '2':
            a = Account()
            a.deposit()

        elif choice == '3':
            a = Account()
            a.withdraw()

        elif choice == '4':
            # 模拟数据生成
            try:
                num = int(input("模拟客户数量 (默认100): ") or "100")
                days = int(input("模拟天数 (默认90): ") or "90")
                X, y = credit_system.simulate_client_behavior(num, days)
                print(f"✅ 已生成{len(X)}条客户数据")
            except Exception as e:
                print(f"❌ 模拟失败: {e}")

        elif choice == '5':
            # 训练模型
            if 'X' not in locals() or 'y' not in locals():
                print("⚠️  未找到数据，正在生成模拟数据...")
                X, y = credit_system.simulate_client_behavior(100, 90)

            credit_system.train_model(X, y)
            print("✅ 模型训练完成！")

        elif choice == '6':
            # 信用风险预测
            if credit_system.model is None:
                print("⚠️  请先训练模型（选择5）")
                continue

            print("\n🔮 信用风险预测")
            print("请输入客户特征：")

            # 示例：生成一个模拟客户
            sample_client = {
                'age': random.randint(25, 55),
                'income': random.randint(30000, 120000),
                'job_stability': random.randint(3, 10),
                'credit_history': random.randint(1, 15),
                'existing_debt_ratio': random.random() * 0.5,
                'initial_balance': random.randint(5000, 50000),
                'deposit_count': random.randint(5, 50),
                'withdrawal_count': random.randint(5, 50),
                'avg_transaction': random.randint(500, 5000),
                'balance_volatility': random.random() * 1000,
                'low_balance_ratio': random.random() * 0.3,
                'final_balance': random.randint(1000, 60000)
            }

            print("📋 示例客户特征：")
            for key, value in sample_client.items():
                print(f"  {key}: {value}")

            use_sample = input("\n使用此示例客户？(y/n): ").lower() == 'y'

            if use_sample:
                result = credit_system.predict_credit_risk(sample_client)
            else:
                # 手动输入特征
                features = []
                for col in credit_system.feature_columns:
                    val = float(input(f"{col}: "))
                    features.append(val)
                result = credit_system.predict_credit_risk(features)

            if result:
                print("\n📈 信用评估结果：")
                for key, value in result.items():
                    print(f"  {key}: {value}")

        elif choice == '7':
            # 系统统计
            print("\n📊 系统统计信息")
            print(f"注册客户数: {len(database)}")
            print(f"总交易记录数: {len(transactions)}")

            if database:
                total_balance = sum([data['basic_info']['balance'] for data in database.values()])
                avg_balance = total_balance / len(database)
                print(f"系统总余额: ¥{total_balance:.2f}")
                print(f"客户平均余额: ¥{avg_balance:.2f}")

        elif choice == '8':
            credit_system.save_model()

        elif choice == '9':
            print("\n感谢使用智能银行系统！")
            break

        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()