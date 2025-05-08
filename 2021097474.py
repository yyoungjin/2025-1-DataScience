import sys
import itertools

class Apriori:
    def __init__(self, min_support_percent):
        self.min_support_percent = min_support_percent # 최소 지지도
        self.transactions = [] # 트랜젝션 데이터 (.txt 파일)
        self.frequent_itemsets = {} # 빈발 아이템셋
        self.rules = []
        
    def load_transactions(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                self.transactions.append(set(items))
        return self.transactions
    
    def get_min_support_num(self):
        return int((self.min_support_percent / 100) * len(self.transactions))
    
    def get_frequent_itemsets(self):
        itemsets = {} # key: 아이템셋, value: 지지도 카운트
        k = 1 
        min_support_count = self.get_min_support_num()
        
        # 1-itemset 생성 (C1)
        for transaction in self.transactions:
            for item in transaction:
                itemset = frozenset([item])
                if itemset not in itemsets:
                    itemsets[itemset] = 0
                itemsets[itemset] += 1
        
        # 빈발 아이템셋 더이상 생성되지 않을 때까지 반복 (Lk -> Ck+1 -> Lk+1)
        while itemsets:
            LK = {}
            for itemset, count in itemsets.items():
                if count >= min_support_count: # 최소 지지도 이상인 아이템셋만 저장
                    LK[itemset] = count
            
            if not LK: # 종료조건
                break
                
            # 현재 빈발 아이템셋 갱신
            self.frequent_itemsets.update(LK)
            
            # 다음 단계 후보 생성
            CKplus = {}
            curr_items = list(LK.keys()) # LK 빈발 아이템셋 리스트
            
            for i in range(len(curr_items)): 
                for j in range(i + 1, len(curr_items)):
                    union_set = curr_items[i] | curr_items[j] # 합집합 연산 (조합)
                    if len(union_set) == k + 1: 
                        CKplus[union_set] = 0 # LK+1 후보 생성
            
            # 후보 개수 세기
            for transaction in self.transactions:
                for candidate in CKplus.keys():
                    if candidate.issubset(transaction):
                        CKplus[candidate] += 1
            
            itemsets = CKplus
            k += 1
            
        return self.frequent_itemsets
    
    def generate_rules(self):
        total_transactions_num = len(self.transactions)
        
        for itemset in self.frequent_itemsets.keys():
            if len(itemset) < 2:
                continue
                
            itemset_support = self.frequent_itemsets[itemset] / total_transactions_num * 100
            
            for i in range(1, len(itemset)):
                for A in itertools.combinations(itemset, i):
                    A = frozenset(A)
                    B = itemset - A
                    confidence = self.frequent_itemsets[itemset] / self.frequent_itemsets[A] * 100
                    self.rules.append((A, B, itemset_support, confidence))
        
        return self.rules
    
    def write_output(self, output_file):
        with open(output_file, 'w') as f:
            for A, B, support, confidence in self.rules:
                A_str = '{' + ','.join(sorted(A)) + '}'
                B_str = '{' + ','.join(sorted(B)) + '}'
                f.write(f'{A_str}\t{B_str}\t{support:.2f}\t{confidence:.2f}\n')

def main():
    if len(sys.argv) != 4:
        print("Usage: python 2021097474.py <min_support(%)> <input_file> <output_file>")
        return
    
    min_support_percent = float(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    apriori = Apriori(min_support_percent)
    apriori.load_transactions(input_file)
    apriori.get_frequent_itemsets()
    apriori.generate_rules()
    apriori.write_output(output_file)

if __name__ == '__main__':
    main()