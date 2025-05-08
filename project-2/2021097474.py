import sys
import math
from collections import Counter

class DecisionTreeNode:
    """결정 트리의 노드를 표현하는 클래스"""
    def __init__(self, is_leaf=False, label=None, attribute=None):
        self.is_leaf = is_leaf  # 리프 노드 여부
        self.label = label      # 예측 클래스 (리프 노드인 경우)
        self.attribute = attribute  # 분할에 사용된 속성
        self.children = {}      # 자식 노드 (속성값 → 노드)

    def __str__(self):
        if self.is_leaf:
            return f"Leaf: {self.label}"
        return f"Node: {self.attribute}"

class DecisionTree:
    """결정 트리 분류기"""
    def __init__(self):
        self.root = None  # 루트 노드
        self.attr_names = None  # 속성 이름 목록
        self.class_name = None  # 클래스 속성 이름
    
    def fit(self, data, header):
        """결정 트리 학습"""
        self.attr_names = header[:-1]  # 마지막은 클래스 속성
        self.class_name = header[-1]
        
        # 속성 이름에서 속성 목록 추출 (마지막 클래스 제외)
        attributes = [name for name in header[:-1]]
        
        # 데이터에서 X(입력 속성들)와 y(클래스) 분리 -> X: 속성들, y: 클래스
        X = [row[:-1] for row in data]
        y = [row[-1] for row in data]
        
        # 트리 생성
        self.root = self._build_tree(X, y, attributes)
        
    def _build_tree(self, X, y, attributes):
        """재귀적으로 결정 트리 생성"""
        # 모든 샘플이 같은 클래스인 경우 (종료 조건 1)
        if len(set(y)) == 1:
            return DecisionTreeNode(is_leaf=True, label=y[0])
        
        # 속성이 없거나, 같은 속성값 조합이 서로 다른 클래스를 가지는 경우 (종료 조건 2)
        if not attributes or self._all_same_attributes(X):
            majority_class = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(is_leaf=True, label=majority_class)
        
        # 최적의 속성 선택 (gain ratio 기준)
        best_attr_idx = self._select_best_attribute(X, y, attributes)
        best_attr = attributes[best_attr_idx]
        
        # 새로운 의사결정 노드 생성
        node = DecisionTreeNode(attribute=best_attr)
        
        # 선택된 속성을 제외한 새로운 속성 리스트 
        new_attributes = list(attributes) # 리스트 복사
        new_attributes.pop(best_attr_idx)
        
        # 각 속성값에 대한 서브트리 생성
        attr_values = self._get_unique_values(X, best_attr_idx)
        
        for value in attr_values:
            # 해당 속성값을 가진 샘플들 찾기
            indices = [i for i, x in enumerate(X) if x[best_attr_idx] == value]
            
            # 해당 서브셋이 비어있으면 다수결로 리프 노드 생성
            if not indices:
                majority_class = Counter(y).most_common(1)[0][0]
                node.children[value] = DecisionTreeNode(is_leaf=True, label=majority_class)
            else:
                # 서브셋 생성
                sub_X = [X[i] for i in indices]
                sub_y = [y[i] for i in indices]
                
                # 재귀적으로 서브트리 구축
                node.children[value] = self._build_tree(sub_X, sub_y, new_attributes)
        
        return node
    
    def _all_same_attributes(self, X):
        """모든 샘플이 같은 속성값을 가지는지 확인"""
        if not X:
            return True
        first = X[0]
        # 모든 샘플의 속성값이 첫 번째 샘플과 동일한지 확인
        for x in X:
            if x != first:
                return False
        return True
    
    def _get_unique_values(self, X, attr_idx):
        """특정 속성의 유일한 값들 반환"""
        return set(x[attr_idx] for x in X)
    
    def _select_best_attribute(self, X, y, attributes) -> int:
        """gain ratio가 가장 높은 속성 선택"""
        n_features = len(attributes)
        
        max_gain_ratio = -float('inf')
        best_attr_idx = -1
        
        for i in range(n_features):
            gain_ratio = self._gain_ratio(X, y, i)
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                best_attr_idx = i
        
        # print("best_attr_idx", best_attr_idx)
        return best_attr_idx
    
    def _entropy(self, y) -> float:
        """엔트로피 계산"""
        n = len(y)
        if n == 0:
            return 0
        
        counts = Counter(y)
        # print("counts", counts) # 각 클래스의 개수
        # 엔트로피 계산
        entropy = 0
        for count in counts.values():
            prob = count / n
            entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _info_gain(self, X, y, attr_idx):
        """정보 이득 계산"""
        n = len(y)
        
        # 전체 엔트로피
        parent_entropy = self._entropy(y)
        
        # 속성으로 분할 후 가중 엔트로피
        values = self._get_unique_values(X, attr_idx)
        # print("values", values)
        weighted_entropy = 0
        
        for value in values:
            indices = [i for i, x in enumerate(X) if x[attr_idx] == value]
            subset_y = [y[i] for i in indices]
            
            weight = len(subset_y) / n
            weighted_entropy += weight * self._entropy(subset_y)
        
        # 정보 이득 = 분할 전 엔트로피 - 분할 후 가중 엔트로피
        info_gain = parent_entropy - weighted_entropy
        # print("info_gain", info_gain)
        return info_gain
    
    def _split_info(self, X, attr_idx):
        """분할 정보 계산 (split information)"""
        n = len(X)
        values = self._get_unique_values(X, attr_idx)
        
        split_info = 0
        for value in values:
            count = sum(1 for x in X if x[attr_idx] == value)
            proportion = count / n
            if proportion > 0:
                split_info -= proportion * math.log2(proportion)
        
        return split_info
    
    def _gain_ratio(self, X, y, attr_idx):
        """이득 비율 계산 (gain ratio)"""
        info_gain = self._info_gain(X, y, attr_idx)
        split_info = self._split_info(X, attr_idx)
        
        # 분모가 0인 경우 처리
        if split_info == 0:
            return 0
        
        return info_gain / split_info
    
    def predict(self, X):
        """여러 샘플에 대한 예측 수행"""
        return [self.predict_one(x) for x in X]
    
    def predict_one(self, x):
        """단일 샘플에 대한 예측 수행"""
        node = self.root
        
        while not node.is_leaf:
            attr_idx = self.attr_names.index(node.attribute)
            attr_value = x[attr_idx]
            
            # 트리에 없는 속성값 처리
            if attr_value not in node.children:
                # 가장 일반적인 자식 선택
                counts = Counter(y for _, y in self.get_node_distribution(node))
                node = DecisionTreeNode(is_leaf=True, label=counts.most_common(1)[0][0])
            else:
                node = node.children[attr_value]
        
        return node.label
    
    def get_node_distribution(self, node):
        """노드에서의 클래스 분포 반환 (디버깅용)"""
        if node.is_leaf:
            return [(None, node.label)]
        
        result = []
        for value, child in node.children.items():
            if child.is_leaf:
                result.append((value, child.label))
            else:
                sub_dist = self.get_node_distribution(child)
                result.extend([(value + "->" + sub_val if sub_val else value, label) 
                               for sub_val, label in sub_dist])
        return result

def read_data(filename):
    """파일에서 데이터 읽기"""
    try:
        # 먼저 입력된 경로 그대로 파일 열기 시도
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # 파일을 찾지 못하면 datasets 폴더 내에서 시도
        try:
            with open(f"datasets/{filename}", 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            # 상대 경로 시도
            with open(f"./datasets/{filename}", 'r') as f:
                lines = f.readlines()
    
    # 헤더(속성 이름) 읽기
    header = lines[0].strip().split('\t')
    
    # 데이터 읽기
    data = []
    for line in lines[1:]:
        if line.strip():  # 빈 줄 무시
            data.append(line.strip().split('\t'))
    
    return data, header

def write_results(filename, test_data, test_header, predictions, class_name):
    """예측 결과를 파일에 쓰기"""
    try:
        # 먼저 입력된 경로 그대로 파일 쓰기 시도
        with open(filename, 'w') as f:
            # 헤더 쓰기 (테스트 속성 + 클래스 이름)
            f.write('\t'.join(test_header + [class_name]) + '\n')
            
            # 데이터와 예측값 쓰기
            for i, row in enumerate(test_data):
                f.write('\t'.join(row + [predictions[i]]) + '\n')
    except:
        # 파일 쓰기에 실패하면 현재 디렉토리에 시도
        with open(f"./{filename}", 'w') as f:
            # 헤더 쓰기 (테스트 속성 + 클래스 이름)
            f.write('\t'.join(test_header + [class_name]) + '\n')
            
            # 데이터와 예측값 쓰기
            for i, row in enumerate(test_data):
                f.write('\t'.join(row + [predictions[i]]) + '\n')

def main():
    """메인 실행 함수"""
    if len(sys.argv) != 4:
        print("Usage: python studentID.py <train_file> <test_file> <result_file>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]
    
    # 학습 데이터 읽기
    train_data, train_header = read_data(train_file)
    # print("train_data", train_data)
    # print("train_header", train_header)
    
    # 테스트 데이터 읽기
    test_data, test_header = read_data(test_file)
    
    # 결정 트리 학습
    tree = DecisionTree()
    tree.fit(train_data, train_header)
    
    # 테스트 데이터 예측
    predictions = tree.predict(test_data)
    
    # 결과 파일 쓰기
    write_results(result_file, test_data, test_header, predictions, train_header[-1])

if __name__ == "__main__":
    main()
