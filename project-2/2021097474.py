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
        self.class_distribution = None  # 클래스 분포 저장

    def __str__(self):
        if self.is_leaf:
            return f"Leaf: {self.label}"
        return f"Node: {self.attribute}"

class DecisionTree:
    """결정 트리 분류기"""
    def __init__(self, max_depth=None, min_samples_split=2):
        self.root = None  # 루트 노드
        self.attr_names = None  # 속성 이름 목록
        self.class_name = None  # 클래스 속성 이름
        self.max_depth = max_depth  # 최대 트리 깊이 (None=무제한)
        self.min_samples_split = min_samples_split  # 분할에 필요한 최소 샘플 수
        self.classes = None  # 가능한 클래스 목록 (알파벳 순)
    
    def fit(self, data, header):
        """결정 트리 학습"""
        self.attr_names = header[:-1]  # 마지막은 클래스 속성
        self.class_name = header[-1]
        
        # 데이터에서 X(입력 속성들)와 y(클래스) 분리
        X = [row[:-1] for row in data]
        y = [row[-1] for row in data]
        
        # 가능한 클래스들을 알파벳 순서로 저장 (일관성을 위해)
        self.classes = sorted(set(y))
        
        # 트리 생성
        self.root = self._build_tree(X, y, self.attr_names.copy(), depth=0)
        
    def _build_tree(self, X, y, attributes, depth=0):
        """재귀적으로 결정 트리 생성"""
        # 노드에 클래스 분포 저장을 위한 카운터 생성
        class_counts = Counter(y)
        
        # 종료 조건 1: 모든 샘플이 같은 클래스
        if len(class_counts) == 1:
            node = DecisionTreeNode(is_leaf=True, label=y[0])
            node.class_distribution = class_counts
            return node
        
        # 종료 조건 2: 최대 깊이 도달
        if self.max_depth is not None and depth >= self.max_depth:
            # 다수결로 클래스 결정 (동점시 알파벳 순)
            majority_class = self._get_majority_class(class_counts)
            node = DecisionTreeNode(is_leaf=True, label=majority_class)
            node.class_distribution = class_counts
            return node
        
        # 종료 조건 3: 속성이 없거나 모든 속성값이 같음
        if not attributes or self._all_same_attributes(X):
            majority_class = self._get_majority_class(class_counts)
            node = DecisionTreeNode(is_leaf=True, label=majority_class)
            node.class_distribution = class_counts
            return node
        
        # 종료 조건 4: 최소 샘플 수 미달
        if len(y) < self.min_samples_split:
            majority_class = self._get_majority_class(class_counts)
            node = DecisionTreeNode(is_leaf=True, label=majority_class)
            node.class_distribution = class_counts
            return node
        
        # 최적의 속성 선택 (gain ratio 기준)
        best_attr_idx = self._select_best_attribute(X, y, attributes)
        
        # 유효한 속성을 찾지 못한 경우
        if best_attr_idx == -1:
            majority_class = self._get_majority_class(class_counts)
            node = DecisionTreeNode(is_leaf=True, label=majority_class)
            node.class_distribution = class_counts
            return node
            
        best_attr = attributes[best_attr_idx]
        best_attr_col_idx = self.attr_names.index(best_attr)  # X 데이터에서의 열 인덱스
        
        # 새로운 의사결정 노드 생성
        node = DecisionTreeNode(attribute=best_attr)
        node.class_distribution = class_counts
        
        # 선택된 속성을 제외한 새로운 속성 리스트
        new_attributes = list(attributes)
        new_attributes.pop(best_attr_idx)
        
        # 각 속성값에 대한 서브트리 생성 (정렬된 순서로 처리)
        attr_values = sorted(set(x[best_attr_col_idx] for x in X))
        
        for value in attr_values:
            # 해당 속성값을 가진 샘플들 찾기
            indices = [i for i, x in enumerate(X) if x[best_attr_col_idx] == value]
            
            # 해당 서브셋이 비어있으면 다수결로 리프 노드 생성
            if not indices:
                majority_class = self._get_majority_class(class_counts)
                leaf = DecisionTreeNode(is_leaf=True, label=majority_class)
                leaf.class_distribution = class_counts
                node.children[value] = leaf
            else:
                # 서브셋 생성
                sub_X = [X[i] for i in indices]
                sub_y = [y[i] for i in indices]
                
                # 재귀적으로 서브트리 구축
                node.children[value] = self._build_tree(sub_X, sub_y, new_attributes, depth+1)
        
        return node
    
    def _get_majority_class(self, counter):
        """다수결 클래스 반환 (동점일 경우 알파벳 순서로 첫 번째 선택)"""
        # 가장 많은 개수를 가진 클래스들 찾기
        max_count = counter.most_common(1)[0][1] if counter else 0
        max_classes = [cls for cls, count in counter.items() if count == max_count]
        
        # 동점이면 알파벳 순 첫 번째 반환
        return sorted(max_classes)[0] if max_classes else (self.classes[0] if self.classes else None)
    
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
    
    def _get_unique_values(self, X, attr_col_idx):
        """특정 속성의 유일한 값들을 정렬해서 반환"""
        return sorted(set(x[attr_col_idx] for x in X))
    
    def _select_best_attribute(self, X, y, attributes):
        """gain ratio가 가장 높은 속성 선택"""
        n_features = len(attributes)
        
        max_gain_ratio = 0.0  # 0 이하는 고려하지 않음
        best_attr_idx = -1
        
        for i in range(n_features):
            attr_name = attributes[i]
            attr_col_idx = self.attr_names.index(attr_name)  # X 데이터에서의 열 인덱스
            
            # 정보 이득과 분할 정보 계산
            info_gain = self._info_gain(X, y, attr_col_idx)
            split_info = self._split_info(X, attr_col_idx)
            
            # 분할 정보가 0이거나 매우 작은 경우 건너뛰기
            if split_info < 0.01:
                continue
            
            # 이득 비율 계산
            gain_ratio = info_gain / split_info
            
            # 최소 정보 이득 임계값 검사 (노이즈 방지)
            if info_gain < 0.01:
                continue
            
            # 최대 이득 비율 갱신
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                best_attr_idx = i
        
        return best_attr_idx
    
    def _entropy(self, y) -> float:
        """엔트로피 계산"""
        n = len(y)
        if n == 0:
            return 0
        
        counts = Counter(y)
        # 엔트로피 계산
        entropy = 0
        for count in counts.values():
            prob = count / n
            entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _info_gain(self, X, y, attr_col_idx):
        """정보 이득 계산"""
        n = len(y)
        
        # 전체 엔트로피
        parent_entropy = self._entropy(y)
        
        # 속성으로 분할 후 가중 엔트로피
        values = self._get_unique_values(X, attr_col_idx)
        weighted_entropy = 0
        
        for value in values:
            indices = [i for i, x in enumerate(X) if x[attr_col_idx] == value]
            subset_y = [y[i] for i in indices]
            
            weight = len(subset_y) / n
            subset_entropy = self._entropy(subset_y)
            weighted_entropy += weight * subset_entropy
        
        # 정보 이득 = 분할 전 엔트로피 - 분할 후 가중 엔트로피
        info_gain = parent_entropy - weighted_entropy
        return info_gain
    
    def _split_info(self, X, attr_col_idx):
        """분할 정보 계산 (split information)"""
        n = len(X)
        values = self._get_unique_values(X, attr_col_idx)
        
        split_info = 0
        for value in values:
            count = sum(1 for x in X if x[attr_col_idx] == value)
            proportion = count / n
            if proportion > 0:
                split_info -= proportion * math.log2(proportion)
        
        return split_info
    
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
                # 현재 노드의 클래스 분포를 기반으로 예측
                if node.class_distribution:
                    return self._get_majority_class(node.class_distribution)
                else:
                    # 클래스 분포가 없는 경우 기본값 반환
                    return self.classes[0] if self.classes else None
            else:
                node = node.children[attr_value]
        
        return node.label

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
    
    # 테스트 데이터 읽기
    test_data, test_header = read_data(test_file)
    
    # 결정 트리 학습 (최대 깊이 및 최소 샘플 수 설정)
    # 자동차 데이터셋은 속성이 6개이므로 깊이 10은 충분함
    tree = DecisionTree(max_depth=10, min_samples_split=2)
    tree.fit(train_data, train_header)
    
    # 테스트 데이터 예측
    predictions = tree.predict(test_data)
    
    # 결과 파일 쓰기
    write_results(result_file, test_data, test_header, predictions, train_header[-1])

if __name__ == "__main__":
    main()
