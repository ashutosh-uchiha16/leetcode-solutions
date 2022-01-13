import com.sun.source.tree.Tree;

import java.util.*;

public class Medium {
    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;

        int sum = 0;
        int carry = 0;
        ListNode result = new ListNode(0);
        ListNode curr = result;

        while (l1 != null || l2 != null) {
            sum = (l1 != null ? l1.val : 0) + (l2 != null ? l2.val : 0) + carry;

            carry = sum / 10;

            result.next = new ListNode(sum % 10);
            result = result.next;

            if (l1 != null)
                l1 = l1.next;
            if (l2 != null)
                l2 = l2.next;

        }

        if (carry > 0)
            result.next = new ListNode(carry);

        return curr.next;
    }

    public int lengthOfLongestSubstring(String s) {
        // sliding window problem
//        int i = 0, j = 0;
//        int max = 0;
//        HashSet<Character> set = new HashSet<>();
//
//        while(j < s.length()){
//            if(!set.contains(s.charAt(j))){
//                set.add(s.charAt(j));
//                j++;
//                max = Math.max(max, set.size());
//            } else {
//                set.remove(s.charAt(i));
//                i++;
//            }
//        }
//
//        return max;

        //Map
        int ans = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        int i = 0;
        for (int j = 0; j < s.length(); j++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }
            ans = Math.max(ans, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }
        return ans;
    }

    public static int myAtoi(String s) {
        if (s.isEmpty())
            return 0;
        int i = 0, sign = 1, base = 0;
        while (i < s.length() && s.charAt(i) == ' ')
            i++;

        if (i >= s.length()) return 0;
        if (s.charAt(i) == '-' || s.charAt(i) == '+')
            sign = s.charAt(i++) == '-' ? -1 : 1;

        while (i < s.length() && s.charAt(i) >= '0' && s.charAt(i) <= '9') {

            if (base > Integer.MAX_VALUE / 10 || (base == Integer.MAX_VALUE / 10 && s.charAt(i) - '0' > 7))
                return (sign == 1) ? Integer.MAX_VALUE : Integer.MIN_VALUE;

            base = base * 10 + (s.charAt(i++) - '0');
        }

        return base * sign;

    }

    public int expandFromMiddle(String s, int left, int right) {
        if (s == null || left > right)
            return 0;

        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }

        return right - left - 1;
    }

    public String longestPalindrome(String s) {
        //O(n) time and O(1) space approach
//        if(s == null || s.length() < 1)
//            return "";
//        int start = 0;
//        int end = 0;
//
//        for(int i = 0; i < s.length(); i++){
//            int len1 = expandFromMiddle(s, i, i);
//            int len2 = expandFromMiddle(s, i, i+1);
//            int len = Math.max(len1, len2);
//            if(len > end - start){
//                start = i - ((len - 1)/2);
//                end = i + (len/2);
//            }
//        }
//        return s.substring(start, end + 1);

        //DP
        int n = s.length();
        String res = null;
        int palindromeStartsAt = 0, maxLen = 0;

        boolean[][] dp = new boolean[n][n];

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < i; j++) {
                dp[i][j] = (s.charAt(i) == s.charAt(j)) && (j - i < 3 || dp[i + 1][j - 1]);

                if (dp[i][j] && (j - i + 1) > maxLen) {
                    palindromeStartsAt = i;
                    maxLen = j - i + 1;
                }
            }
        }

        return s.substring(palindromeStartsAt, palindromeStartsAt + maxLen);


    }

    public static String convert(String s, int numRows) {
        char[] c = s.toCharArray();
        int len = c.length;
        StringBuffer[] sb = new StringBuffer[numRows];
        for (int i = 0; i < sb.length; i++) {
            sb[i] = new StringBuffer();
        }
        int i = 0;
        while (i < len) {
            for (int idx = 0; idx < numRows && i < len; idx++) {
                sb[idx].append(c[i++]);
            }

            for (int idx = numRows - 2; idx >= 1 && i < len; idx--) {
                sb[idx].append(c[i++]);
            }
        }
        for (int idx = 1; idx < sb.length; idx++) {
            sb[0].append(sb[idx]);
        }

        return sb[0].toString();
    }

    public static int maxArea(int[] height) {
        int maxArea = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            maxArea = Math.max(maxArea, Math.min(height[left], height[right]) * (right - left));
            if (height[left] < height[right])
                left++;
            else if (height[left] == height[right]) {
                left++;
                right--;
            } else
                right--;
        }

        return maxArea;

    }

    public String intToRoman(int num) {
        TreeMap<Integer, String> map = new TreeMap<>();
        map.put(1000, "M");
        map.put(900, "CM");
        map.put(500, "D");
        map.put(400, "CD");
        map.put(100, "C");
        map.put(90, "XC");
        map.put(50, "L");
        map.put(40, "XL");
        map.put(10, "X");
        map.put(9, "IX");
        map.put(5, "V");
        map.put(4, "IV");
        map.put(1, "I");

        StringBuilder sb = new StringBuilder();
        while (num != 0) {
            int floor = map.floorKey(num);
            sb.append(map.get(floor));
            num -= floor;
        }
        return sb.toString();
    }

    public static void dfs(int s, String digits, StringBuilder sb, List<String> res, Map<Character, String> map) {

        if (s == digits.length()) {
            res.add(sb.toString());
            return;
        }

        String str = map.get(digits.charAt(s));
        for (int i = 0; i < str.length(); i++) {
            sb.append(str.charAt(i));
            dfs(s + 1, digits, sb, res, map);
            sb.setLength(sb.length() - 1);
        }
    }

    public static List<String> letterCombinations(String digits) {
        if (digits == null || digits.length() == 0)
            return new ArrayList<>();

        Map<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");

        List<String> res = new ArrayList<>();
        dfs(0, digits, new StringBuilder(), res, map);

        return res;
    }

    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new LinkedList<>();
        if (nums.length < 3)
            return ans;
        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 2; i++) {
//            if(nums[i] > 0)
//                break;
            if (i == 0 || (i > 0 && nums[i] != nums[i - 1])) {
                int low = i + 1, high = nums.length - 1, sum = 0 - nums[i];
                while (low < high) {
                    if (nums[low] + nums[high] == sum) {
                        ans.add(Arrays.asList(nums[i], nums[low], nums[high]));
                        while (low < high && nums[low] == nums[low + 1])
                            low++;
                        while (low < high && nums[high] == nums[high - 1])
                            high--;
                        low++;
                        high--;
                    } else if (nums[low] + nums[high] < sum)
                        low++;
                    else
                        high--;
                }
            }

        }

        return ans;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode temp = new ListNode(0);
        temp.next = head;
        ListNode a = temp, b = temp;
        while (n > 0) {
            b = b.next;
            n--;
        }
        while (b.next != null) {
            a = a.next;
            b = b.next;
        }
        a.next = a.next.next;
        return temp.next;
    }

    public void backtrack(List<String> ans, String str, int open, int close, int max) {
        //base case
        if (str.length() == max * 2) {
            ans.add(str);
            return;
        }
        if (open < max)
            backtrack(ans, str + "(", open + 1, close, max);
        if (close < open)
            backtrack(ans, str + ")", open, close + 1, max);

    }

    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        backtrack(ans, "", 0, 0, n);
        return ans;
    }

    public int search(int[] nums, int target) {
        int n = nums.length;
        int low = 0, high = n - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == target)
                return mid;
            if (nums[low] <= nums[mid]) {
                if (nums[low] <= target && target < nums[mid])
                    high = mid - 1;
                else
                    low = mid + 1;
            } else {
                if (nums[mid] < target && target <= nums[high])
                    low = mid + 1;
                else
                    high = mid - 1;
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return new int[]{-1, -1};
        int found = -1;
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                found = mid;
                break;
            } else if (nums[mid] > target)
                right = mid - 1;
            else
                left = mid + 1;
        }
        if (found == -1)
            return new int[]{-1, -1};
        int low = found, high = found;
        for (; low >= 0 && nums[low] == target; low--) ;
        for (; high < nums.length && nums[high] == target; high++) ;

        return new int[]{low + 1, high - 1};
    }

    public void islandsDFS(char[][] grid, int i, int j) {

        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0')
            return;

        grid[i][j] = '0';
        islandsDFS(grid, i - 1, j);
        islandsDFS(grid, i + 1, j);
        islandsDFS(grid, i, j - 1);
        islandsDFS(grid, i, j + 1);
    }

    public int numIslands(char[][] grid) {
        int count = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    islandsDFS(grid, i, j);
                }
            }
        }
        return count;
    }

    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null)
            return head;

        ListNode odd = head;
        ListNode even = head.next;
        ListNode evenHead = even;

        while (even != null && even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms) {

        boolean[] visited = new boolean[rooms.size()];
        visited[0] = true;

        Stack<Integer> stack = new Stack<>();
        stack.push(0);

        while (!stack.isEmpty()) {
            int temp = stack.pop();
            for (int room : rooms.get(temp)) {
                if (!visited[room]) {
                    visited[room] = true;
                    stack.push(room);
                }
            }
        }

        for (boolean v : visited)
            if (!v)
                return false;

        return true;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public void func(TreeNode root, List<Integer> ans, int level) {
        if (root == null)
            return;

        if (level == ans.size())
            ans.add(root.val);
        else
            ans.set(level, Math.max(ans.get(level), root.val));

        func(root.left, ans, level + 1);
        func(root.right, ans, level + 1);
    }

    public List<Integer> largestValue(TreeNode root) {

        List<Integer> ans = new ArrayList<>();
        func(root, ans, 0);
        return ans;
    }

    public TreeNode pruneTree(TreeNode root) {
        if (root == null)
            return null;
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);

        if (root.left == null && root.right == null && root.val == 0)
            return null;

        return root;
    }

    public static boolean isNStraightHand(int[] hand, int groupSize) {

        TreeMap<Integer, Integer> counts = new TreeMap<>();
        for (int card : hand)
            counts.put(card, counts.getOrDefault(card, 0) + 1);
        while (counts.size() > 0) {
            int first = counts.firstKey();
            for (int i = first; i < first + groupSize; i++) {

                if (!counts.containsKey(i))
                    return false;
                int count = counts.get(i);
                if (count == 1)
                    counts.remove(i);
                else
                    counts.put(i, --count);
            }
        }
        return true;
    }

    public ListNode swapPairs(ListNode head) {
        //recursive
//        if(head == null || head.next == null)
//            return head;
//
//        ListNode tmp1 = head;
//        ListNode tmp2 = head.next;
//
//        tmp1.next = tmp2.next;
//        tmp2.next = tmp1;
//
//        tmp1.next = swapPairs(tmp1.next);
//
//        return tmp2;

        //iterative
        ListNode temp = new ListNode(0);
        temp.next = head;
        ListNode curr = temp;

        while (curr != null && curr.next != null) {
            ListNode first = curr.next;
            ListNode sec = curr.next.next;

            first.next = sec.next;
            curr.next = sec;
            sec.next = first;
            curr = curr.next.next;
        }

        return temp.next;
    }

    public int findBottomLeftValue(TreeNode root) {

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        TreeNode temp = null;
        while (!q.isEmpty()) {
            temp = q.poll();
            if (temp.right != null)
                q.offer(temp.right);
            if (temp.left != null)
                q.offer(temp.left);

        }
        return temp.val;
    }

    public ListNode partition(ListNode head, int x) {
        ListNode before = new ListNode(0);
        ListNode after = new ListNode(0);
        ListNode curr1 = before, curr2 = after;

        while (head != null) {
            if (head.val < x) {
                curr1.next = head;
                curr1 = curr1.next;
            } else {
                curr2.next = head;
                curr2 = curr2.next;
            }
            head = head.next;
        }

        curr2.next = null;
        curr1.next = after.next;

        return before.next;
    }

    int res;

    public int distributeCoins(TreeNode root) {
        res = 0;
        dfs(root);
        return res;
    }

    public int dfs(TreeNode root) {

        if (root == null)
            return 0;
        int left = dfs(root.left);
        int right = dfs(root.right);

        res += Math.abs(left) + Math.abs(right);
        return root.val + left + right - 1;
    }

    public boolean isCompleteTree(TreeNode root) {

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (q.peek() != null) {
            TreeNode temp = q.poll();
            q.offer(temp.left);
            q.offer(temp.right);
        }
        while (!q.isEmpty() && q.peek() == null)
            q.poll();

        return q.isEmpty();
    }

    public void flatten(TreeNode root) {
        //iterative way
        if (root == null)
            return;
        Stack<TreeNode> st = new Stack<>();
        st.push(root);

        while (!st.isEmpty()) {
            TreeNode temp = st.pop();
            if (temp.right != null)
                st.push(temp.right);
            if (temp.left != null)
                st.push(temp.left);
            if (!st.isEmpty())
                temp.right = st.peek();

            temp.left = null;
        }

//        if(root == null)
//            return;
//
//        TreeNode left = root.left;
//        TreeNode right = root.right;
//
//        root.left = null;
//
//        flatten(left);
//        flatten(right);
//
//        root.right = left;

    }

    public void rightView(TreeNode root, List<Integer> ans, int depth) {
        if (root == null)
            return;

        if (ans.size() == depth)
            ans.add(root.val);

        rightView(root.right, ans, depth + 1);
        rightView(root.left, ans, depth + 1);
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        rightView(root, ans, 0);
        return ans;
    }

    public ListNode merge(ListNode left, ListNode right) {
        ListNode l = new ListNode(0), curr = l;

        while (left != null && right != null) {
            if (left.val < right.val) {
                curr.next = left;
                left = left.next;
            } else {
                curr.next = right;
                right = right.next;
            }

            curr = curr.next;
        }
        if (left != null)
            curr.next = left;
        if (right != null)
            curr.next = right;

        return l.next;

    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null)
            return head;

        ListNode prev = null;
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }

        prev.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(slow);

        return merge(left, right);
    }

    public List<Integer> spiralOrder(int[][] matrix) {

        List<Integer> ans = new ArrayList<>();
        int top = 0, bottom = matrix.length - 1, left = 0, right = matrix[0].length - 1;
        while (top <= bottom && left <= right) {

            for (int i = left; i <= right; i++)
                ans.add(matrix[top][i]);
            top++;

            for (int i = top; i <= bottom; i++)
                ans.add(matrix[i][right]);
            right--;
            if (top <= bottom) {
                for (int i = right; i >= left; i--)
                    ans.add(matrix[bottom][i]);
                bottom--;
            }
            if (left <= right) {
                for (int i = bottom; i >= top; i--)
                    ans.add(matrix[i][left]);
                left++;
            }
        }
        return ans;
    }

    public int[][] generateMatrix(int n) {
        int[][] ans = new int[n][n];

        int rowBegin = 0, rowEnd = n - 1, colBegin = 0, colEnd = n - 1;
        int counter = 1;

        while (rowBegin <= rowEnd && colBegin <= colEnd) {

            for (int i = colBegin; i <= colEnd; i++)
                ans[rowBegin][i] = counter++;

            rowBegin++;

            for (int i = rowBegin; i <= rowEnd; i++)
                ans[i][colEnd] = counter++;

            colEnd--;

            if (rowBegin <= rowEnd) {
                for (int i = colEnd; i >= colBegin; i--)
                    ans[rowEnd][i] = counter++;

            }
            rowEnd--;
            if (colBegin <= colEnd) {
                for (int i = rowEnd; i >= rowBegin; i--)
                    ans[i][colBegin] = counter++;
            }

            colBegin++;
        }

        return ans;
    }

    public int minSubArrayLen(int target, int[] nums) {

        int sum = 0, min = Integer.MAX_VALUE;
        int j = 0;

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            while (sum >= target) {
                min = Math.min(min, i - j + 1);
                sum -= nums[j++];
            }
        }

        return min == Integer.MAX_VALUE ? 0 : min;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0)
            return false;

        int row = matrix.length;
        int col = matrix[0].length;

        int left = 0, right = row * col - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int mid_ele = matrix[mid / col][mid % col];
            if (mid_ele == target)
                return true;
            else if (mid_ele > target)
                right = mid - 1;
            else
                left = mid + 1;
        }

        return false;
    }

    public void permuteBackTrack(List<List<Integer>> result, int[] nums, List<Integer> currList, int index) {
        if (currList.size() == nums.length) {
            result.add(currList);
            return;
        }

        int n = nums[index];
        for (int i = 0; i <= currList.size(); i++) {
            List<Integer> temp = new ArrayList<>(currList);
            temp.add(i, n);
            permuteBackTrack(result, nums, temp, index + 1);
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();

        permuteBackTrack(result, nums, new ArrayList<Integer>(), 0);

        return result;
    }


    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<>());

        for (int n : nums) {
            int size = result.size();
            for (int i = 0; i < size; i++) {
                List<Integer> subset = new ArrayList<>(result.get(i));
                subset.add(n);
                result.add(subset);
            }
        }

        return result;

    }

    public static int count = 0;
    public static int number = 0;

    public void kthSmallestfunc(TreeNode node) {
        if (node.left != null)
            kthSmallestfunc(node.left);

        count--;

        if (count == 0) {
            number = node.val;
            return;
        }

        if (node.right != null)
            kthSmallestfunc(node.right);
    }

    public int kthSmallest(TreeNode root, int k) {

        count = k;
        kthSmallestfunc(root);

        return number;
    }

    public int areaOfIsland(int[][] grid, int i, int j) {

        if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == 1) {
            grid[i][j] = 0;
            return 1 + areaOfIsland(grid, i + 1, j) + areaOfIsland(grid, i - 1, j) + areaOfIsland(grid, i, j - 1) + areaOfIsland(grid, i, j + 1);
        }
        return 0;
    }

    public int maxAreaOfIsland(int[][] grid) {
        int max_area = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    max_area = Math.max(max_area, areaOfIsland(grid, i, j));
                }
            }
        }

        return max_area;
    }

    public int totalFruit(int[] tree) {
        int last = -1, second_last = -1, last_count = 0;
        int curr_max = 0, max = 0;

        for (int fruit : tree) {

            if (fruit == last || fruit == second_last)
                curr_max++;
            else
                curr_max = last_count + 1;

            if (fruit == last)
                last_count++;
            else {
                last_count = 1;
            }

            if (fruit != last) {
                second_last = last;
                last = fruit;
            }

            max = Math.max(max, curr_max);
        }
        return max;
    }

    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int n = grid.length;
        int[] row = new int[n], col = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                row[i] = Math.max(row[i], grid[i][j]);
                col[j] = Math.max(col[j], grid[i][j]);
            }
        }
        int sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += Math.min(row[i], col[j]) - grid[i][j];
            }
        }
        return sum;
    }

    public int subarraySum(int[] nums, int k) {

        Map<Integer, Integer> map = new HashMap<>();
        int sum = 0, ans = 0;
        map.put(0, 1);

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];

            if (map.containsKey(sum - k))
                ans += map.get(sum - k);

            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }

        return ans;
    }

    public ListNode reverse(ListNode node) {
        ListNode curr = node, prev = null;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }

        return prev;
    }

    public void mergeList(ListNode head1, ListNode head2) {

        while (head1 != null) {
            ListNode tmp1 = head1.next, tmp2 = head2.next;
            head1.next = head2;

            if (tmp1 == null)
                break;

            head2.next = tmp1;
            head1 = tmp1;
            head2 = tmp2;
        }
    }

    public void reorderList(ListNode head) {

        if (head == null || head.next == null)
            return;
        ListNode slow = head, fast = head, endOfFirst = null;
        while (fast != null && fast.next != null) {
            endOfFirst = slow;
            slow = slow.next;
            fast = fast.next.next;
        }

        endOfFirst.next = null;

        ListNode head1 = head;
        ListNode head2 = reverse(slow);

        mergeList(head1, head2);


    }

    public ListNode reverseBetween(ListNode head, int left, int right) {

        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode curr1 = dummy, temp1 = null;

        for (int i = 0; i < left; i++) {
            temp1 = curr1;
            curr1 = curr1.next;
        }

        ListNode curr2 = curr1, temp2 = temp1, prev = null;

        for (int i = left; i <= right; i++) {
            prev = curr2.next;
            curr2.next = temp2;
            temp2 = curr2;
            curr2 = prev;
        }

        temp1.next = temp2;
        curr1.next = curr2;

        return dummy.next;


    }

    public int[] nextLargerNodes(ListNode head) {

        ArrayList<Integer> A = new ArrayList<>();
        ListNode curr = head;

        while (curr != null) {
            A.add(curr.val);
            curr = curr.next;
        }
        int[] ans = new int[A.size()];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < A.size(); i++) {

            while (!stack.isEmpty() && A.get(stack.peek()) < A.get(i))
                ans[stack.pop()] = A.get(i);
            stack.push(i);
        }

        return ans;
    }

    public void reverse(int[] arr, int left, int right) {
        while (left < right) {
            int temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            left++;
            right--;
        }
    }

    public void rotate(int[] nums, int k) {
        k = k % nums.length;

        if (nums.length == 1 || k == 0)
            return;

        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public static String pushDominoes(String dominoes) {

//        char[] chars = dominoes.toCharArray();
//        int n = dominoes.length();
//
//        int l = -1, r = -1;
//        for (int i = 0; i <= n; i++) {
//            if (i == chars.length || chars[i] == 'R') {
//                if (r > l) {
//                    while (r < i)
//                        chars[r++] = 'R';
//                }
//                r = i;
//            } else if (chars[i] == 'L') {
//                if (l > r || (l == -1 && r == -1)) {
//                    while (++l < i)
//                        chars[l] = 'L';
//                } else {
//                    l = i;
//                    int low = r + 1, high = l - 1;
//                    while (low < high) {
//                        chars[low++] = 'R';
//                        chars[high--] = 'L';
//                    }
//                }
//            }
//        }
//
//        return new String(chars);

//        char[] arr = dominoes.toCharArray();
//        int n = arr.length;
//        int[] forces = new int[n];
//
//        int force = 0;
//        for(int i = 0; i < n; i++){
//
//            if(arr[i] == 'R'){
//                force = n;
//            } else if(arr[i] == 'L'){
//                force = 0;
//            } else{
//                force = Math.max(force - 1, 0);
//            }
//            forces[i] += force;
//        }
//
//        force = 0;
//        for(int i = n-1; i >= 0; i--){
//            if(arr[i] == 'L'){
//                force = n;
//            } else if(arr[i] == 'R'){
//                force = 0;
//            } else{
//                force = Math.max(force - 1, 0);
//            }
//            forces[i] -= force;
//        }
//
//
//
//        StringBuilder sb = new StringBuilder();
//
//        for(Integer f: forces){
//            if(f > 0)
//                sb.append('R');
//            else if(f < 0)
//                sb.append('L');
//            else
//                sb.append(".");
//        }
//
//        return sb.toString();

        char[] arr = dominoes.toCharArray();
        int n = arr.length;
        int l = -1, r = -1;

        for(int i = 0; i < n; i++){

            if(i == arr.length || arr[i] == 'R'){

                if( r > l){
                    while(r < i){
                        arr[r++] = 'R';
                    }
                }
                r = i;
            } else if(arr[i] == 'L'){

                if(l > r || (r == -1 && l == -1)){
                    while(++l < i){
                        arr[l] = 'L';
                    }
                } else {
                    l = i;
                    int low = r + 1, high = l - 1;
                    while(low < high){
                        arr[low++] = 'R';
                        arr[high--] = 'L';
                    }
                }
            }
        }

        return new String(arr);
    }

    public void sortColors(int[] nums) {

        if (nums.length == 0 || nums.length == 1)
            return;

        int left = 0, right = nums.length - 1, idx = 0;

        while (idx <= right && left < right) {

            if (nums[idx] == 0) {
                nums[idx] = nums[left];
                nums[left] = 0;
                left++;
                idx++;
            } else if (nums[idx] == 2) {
                nums[idx] = nums[right];
                nums[right] = 2;
                right--;
            } else
                idx++;
        }
    }

    public int deepestLeavesSum(TreeNode root) {
        int ans = 0;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {
            ans = 0;
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode temp = q.poll();
                ans += temp.val;

                if (temp.left != null)
                    q.offer(temp.left);
                if (temp.right != null)
                    q.offer(temp.right);
            }
        }
        return ans;
    }

    int deepest = 0, sum = 0;

    public int deepestLeavesSumDFShelper(TreeNode node, int depth) {
        if (node == null)
            return 0;

        if (node.left == null && node.right == null) {
            if (deepest == depth)
                sum += node.val;
            else if (depth > deepest) {
                sum = node.val;
                deepest = depth;
            }

        }

        deepestLeavesSumDFShelper(node.left, depth + 1);
        deepestLeavesSumDFShelper(node.right, depth + 1);

        return sum;
    }

    public int deepestLeavesSumDFS(TreeNode root) {
        return deepestLeavesSumDFShelper(root, 0);
    }

    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        if (original == null || original == target)
            return cloned;

        TreeNode res = getTargetCopy(original.left, cloned.left, target);
        if (res != null)
            return res;
        return getTargetCopy(original.right, cloned.right, target);

    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {

        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        if (root == null)
            return ans;
        Stack<TreeNode> s1 = new Stack<>();
        Stack<TreeNode> s2 = new Stack<>();
        s1.push(root);
        while (!s1.isEmpty() || !s2.isEmpty()) {

            List<Integer> lst = new ArrayList<>();
            while (!s1.isEmpty()) {
                TreeNode temp = s1.pop();
                lst.add(temp.val);
                if (temp.left != null)
                    s2.push(temp.left);
                if (temp.right != null)
                    s2.push(temp.right);
            }
            ans.add(lst);
            lst = new ArrayList<>();
            while (!s2.isEmpty()) {
                TreeNode temp = s2.pop();
                lst.add(temp.val);
                if (temp.right != null)
                    s1.push(temp.right);
                if (temp.left != null)
                    s1.push(temp.left);
            }
            if (!lst.isEmpty())
                ans.add(lst);
        }
        return ans;
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {

        Queue<TreeNode> q = new LinkedList<>();
        List<List<Integer>> ans = new ArrayList<List<Integer>>();

        if (root == null)
            return ans;
        q.offer(root);

        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> lst = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode temp = q.poll();
                if (temp.left != null)
                    q.offer(temp.left);
                if (temp.right != null)
                    q.offer(temp.right);
                lst.add(temp.val);
            }
            ans.add(0, lst);
        }
        return ans;
    }
//    class Node {
//        public int val;
//        public Node left;
//        public Node right;
//        public Node next;
//
//        public Node() {}
//
//        public Node(int _val) {
//            val = _val;
//        }
//
//        public Node(int _val, Node _left, Node _right, Node _next) {
//            val = _val;
//            left = _left;
//            right = _right;
//            next = _next;
//        }
//    };
//    public Node connect(Node root){
//       Node head = null, prev = null, curr = root;
//
//       while(curr != null){
//
//           while(curr != null){
//
//               if(curr.left != null){
//                   if(prev != null){
//                       prev.next = curr.left;
//                   } else
//                       head = curr.left;
//                   prev = curr.left;
//               }
//
//               if(curr.right != null){
//                   if(prev != null){
//                       prev.next = curr.right;
//                   } else
//                       head = curr.right;
//                   prev = curr.right;
//               }
//               curr = curr.next;
//           }
//           curr = head;
//           prev = null;
//           head = null;
//       }
//
//
//    }

    public int threeSumClosest(int[] nums, int target) {

        int diff = Integer.MAX_VALUE, n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            int low = i + 1, high = n - 1;
            while (low < high) {
                int sum = nums[i] + nums[low] + nums[high];
                if (Math.abs(sum - target) < Math.abs(diff))
                    diff = target - sum;
                if (sum < target)
                    ++low;
                else
                    --high;
            }

        }
        return target - diff;

    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reversePer(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    public void nextPermutation(int[] nums) {

        if (nums == null || nums.length <= 1)
            return;

        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1])
            i--;

        if (i >= 0) {
            int j = nums.length - 1;
            while (nums[i] >= nums[j])
                j--;
            swap(nums, i, j);
        }
        reversePer(nums, i + 1);
    }

    public void backtrackCombinations(List<List<Integer>> ans, List<Integer> list, int[] candidates, int remains, int idx) {

        if (remains < 0)
            return;
        else if (remains == 0)
            ans.add(new ArrayList<>(list));
        else {
            for (int i = idx; i < candidates.length; i++) {
                list.add(candidates[i]);
                backtrackCombinations(ans, list, candidates, remains - candidates[i], i);
                list.remove(list.size() - 1);
            }
        }

    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        backtrackCombinations(ans, new ArrayList<>(), candidates, target, 0);

        return ans;
    }

    public void backtrackCombinations2(List<List<Integer>> ans, List<Integer> list, int[] candidates, int remains, int idx) {
        if (remains < 0)
            return;
        else if (remains == 0)
            ans.add(new ArrayList<>(list));
        else {
            for (int i = idx; i < candidates.length; i++) {

                if (i > idx && candidates[i] == candidates[i - 1])
                    continue;
                list.add(candidates[i]);
                backtrackCombinations2(ans, list, candidates, remains - candidates[i], i + 1);
                list.remove(list.size() - 1);
            }
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {

        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        backtrackCombinations2(ans, new ArrayList<>(), candidates, target, 0);

        return ans;
    }

    public void permuteBackTrack2(List<List<Integer>> ans, List<Integer> list, int[] nums, boolean[] used) {
        if (list.size() == nums.length) {
            ans.add(new ArrayList<>(list));
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (used[i] || i > 0 && nums[i] == nums[i - 1] && !used[i - 1])
                    continue;
                used[i] = true;
                list.add(nums[i]);
                permuteBackTrack2(ans, list, nums, used);
                used[i] = false;
                list.remove(list.size() - 1);
            }
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {

        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        permuteBackTrack2(ans, new ArrayList<>(), nums, new boolean[nums.length]);

        return ans;
    }

    public void rotate(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = i; j < matrix[0].length; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][matrix.length - j - 1];
                matrix[i][matrix.length - j - 1] = temp;
            }
        }
    }

    public boolean canJump(int[] nums) {
//        int idx = 0;
//        for(int i = 0; i < nums.length; i++){
//
//            if(i > idx)
//                return false;
//            idx = Math.max(idx, nums[i] + i);
//        }
//
//        return true;
        int n = nums.length, idx = Integer.MIN_VALUE;

        for (int i = 0; i < n; i++) {
            idx = Math.max(idx, nums[i] + i);

            if (idx >= n - 1)
                return true;
            if (idx == i && nums[i] == 0)
                return false;
        }

        return false;
    }

    public int jump(int[] nums) {
        int jumps = 0, idx = 0, idxEnd = 0;

        for (int i = 0; i < nums.length - 1; i++) {
            idx = Math.max(idx, nums[i] + i);

            if (i == idxEnd) {
                jumps++;
                idxEnd = idx;
            }
        }
        return jumps;
    }

    int sum1 = 0;

    public int sumEvenGrandparenthelper(TreeNode curr, TreeNode parent, TreeNode grandparent) {
        if (curr == null)
            return 0;
        if (grandparent != null && grandparent.val % 2 == 0) {
            sum1 += curr.val;
        }
        sumEvenGrandparenthelper(curr.left, curr, parent);
        sumEvenGrandparenthelper(curr.right, curr, parent);

        return sum1;
    }

    public int sumEvenGrandparent(TreeNode root) {

        return sumEvenGrandparenthelper(root, null, null);
    }

    int sum2 = 0;

    public TreeNode bstToGst(TreeNode root) {

        if (root.right != null)
            bstToGst(root.right);
        sum2 = sum2 + root.val;
        if (root.left != null)
            bstToGst(root.left);

        return root;
    }

    public void inorder1(TreeNode node, Queue<Integer> q) {
        if (node == null)
            return;
        inorder1(node.left, q);
        q.offer(node.val);
        inorder1(node.right, q);
    }

    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {


        Queue<Integer> q1 = new LinkedList<>(), q2 = new LinkedList<>();
        List<Integer> ans = new ArrayList<>();

        inorder1(root1, q1);
        inorder1(root2, q2);

        while (!q1.isEmpty() || !q2.isEmpty()) {
            if (q1.isEmpty()) {
                ans.add(q2.poll());
            } else if (q2.isEmpty()) {
                ans.add(q1.poll());
            } else {
                ans.add(q1.peek() < q2.peek() ? q1.poll() : q2.poll());
            }
        }

        return ans;
    }

    List<TreeNode> sortedArray = new ArrayList<>();

    public void inorderTraversal(TreeNode node) {
        if (node == null)
            return;
        inorderTraversal(node.left);
        sortedArray.add(node);
        inorderTraversal(node.right);
    }

    public TreeNode arrayToBST(int left, int right) {
        if (left > right)
            return null;
        int mid = left + (right - left) / 2;
        TreeNode midNode = sortedArray.get(mid);

        midNode.left = arrayToBST(left, mid - 1);
        midNode.right = arrayToBST(mid + 1, right);

        return midNode;
    }

    public TreeNode balanceBST(TreeNode root) {

        inorderTraversal(root);
        return arrayToBST(0, sortedArray.size() - 1);
    }

    public int countVowelStrings(int n) {

//        int[][] dp = new int[n + 1][6];
//        for(int i = 1; i <= n; i++){
//            for(int j = 1; j <= 5; j++){
//                dp[i][j] = dp[i][j - 1] + ((i > 1) ? dp[i - 1][j]: 1);
//            }
//        }
//
//        return dp[n][5];

        int[] dp = new int[]{0, 1, 1, 1, 1, 1};
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= 5; j++) {
                dp[j] = dp[j] + dp[j - 1];
            }
        }
        return dp[5];
    }

    public TreeNode removeLeafNodes(TreeNode root, int target) {

        if (root == null)
            return null;
        root.left = removeLeafNodes(root.left, target);
        root.right = removeLeafNodes(root.right, target);

        if (root.left == null && root.right == null && root.val == target)
            return null;

        return root;
    }

    public int helperfuncDiff(TreeNode root, int min, int max) {
        if (root == null)
            return max - min;

        max = Math.max(max, root.val);
        min = Math.min(min, root.val);

        return Math.max(helperfuncDiff(root.left, min, max), helperfuncDiff(root.right, min, max));

    }

    public int maxAncestorDiff(TreeNode root) {

        return helperfuncDiff(root, root.val, root.val);
    }

    int[] count1 = new int[10];
    int ans = 0;

    public void helper(TreeNode node) {
        if (node == null)
            return;

        count1[node.val]++;
        if (node.left == null && node.right == null) {
            int odd = 0;

            for (int i = 1; i < 10; i++) {

                if (count1[i] % 2 != 0)
                    odd++;

            }
            if (odd < 2)
                ans++;
        }

        helper(node.left);
        helper(node.right);

        count1[node.val]--;
    }

    public int pseudoPalindromicPaths(TreeNode root) {

        helper(root);
        return ans;
    }

    int maxDepth = 0;
    TreeNode res1 = null;

    public int helper2(TreeNode node, int depth) {
        maxDepth = Math.max(maxDepth, depth);

        if (node == null)
            return depth;

        int lDepth = helper2(node.left, depth + 1);
        int rDepth = helper2(node.right, depth + 1);

        if (lDepth == rDepth && lDepth == maxDepth) {
            res1 = node;
        }

        return Math.max(lDepth, rDepth);
    }

    public TreeNode lcaDeepestLeaves(TreeNode root) {
        helper2(root, 0);

        return res1;

    }

    public TreeNode helper3(TreeNode node, List<TreeNode> ans, Set<Integer> del) {

        if (node == null)
            return null;

        node.left = helper3(node.left, ans, del);
        node.right = helper3(node.right, ans, del);

        if (del.contains(node.val)) {
            if (node.left != null) {
                ans.add(node.left);
            }
            if (node.right != null)
                ans.add(node.right);

            return null;
        }

        return node;

    }

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {

        List<TreeNode> ans = new ArrayList<>();

        Set<Integer> del = new HashSet<>();

        for (int i : to_delete)
            del.add(i);

        helper3(root, ans, del);

        if (!del.contains(root.val)) {
            ans.add(root);
        }
        return ans;
    }

    public boolean flipEquiv(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null)
            return root1 == root2;

        return (root1.val == root2.val) && (flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left) || flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right));
    }

    public TreeNode trimBST(TreeNode root, int low, int high) {

        if(root == null)
            return null;

        if(root.val < low)
            return trimBST(root.right, low, high);

        if(root.val > high)
            return trimBST(root.left, low, high);

        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);

        return root;
    }

    public int countSquares(int[][] matrix) {

        int ans = 0;

        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){

                if(matrix[i][j] > 0 && i > 0 && j > 0){
                    matrix[i][j] = Math.min(matrix[i-1][j-1], Math.min(matrix[i-1][j], matrix[i][j-1])) + 1;
                }
                ans += matrix[i][j];
            }
        }
        return ans;
    }

    public int numTeams(int[] rating){

        int count = 0, n = rating.length;
        for(int j = 0; j < n; j++){
            int leftSmaller = 0, leftLarger = 0;
            int rightSmaller = 0, rightLarger = 0;

            for(int i = 0; i < j; i++){

                if(rating[i] < rating[j])
                    leftSmaller++;
                else
                    leftLarger++;

            }
            for(int k = j + 1; k < n; k++){

                if(rating[j] < rating[k])
                    rightLarger++;
                else
                    rightSmaller++;

            }
            count += (leftSmaller * rightLarger) + (leftLarger * rightSmaller);
        }

        return count;
    }

    public int numSplits(String s){

//        Map<Character, Integer> left = new HashMap<>();
//        Map<Character, Integer> right = new HashMap<>();
//
//        for(int i = 0; i < s.length(); i++)
//            right.put(s.charAt(i), right.getOrDefault(s.charAt(i), 0) + 1);
//
//        int count = 0;
//
//        for(int i = 0; i < s.length(); i++){
//
//            char ch = s.charAt(i);
//
//            left.put(s.charAt(i), left.getOrDefault(s.charAt(i), 0) + 1);
//
//            right.put(s.charAt(i), right.getOrDefault(s.charAt(i), 0) - 1);
//
//            if(right.get(ch) <= 0)
//                right.remove(ch);
//
//            if(left.size() == right.size())
//                count++;
//        }
//
//        return count;

        int[] leftArr = new int[26], rightArr = new int[26];
        int count = 0, left = 0, right = 0;

        for(char c: s.toCharArray()){

            if(rightArr[c - 'a'] == 0)
                right++;
            rightArr[c - 'a']++;
        }

        for(char c: s.toCharArray()){

            if(leftArr[c - 'a'] == 0)
                left++;
            leftArr[c - 'a']++;

            rightArr[c - 'a']--;
            if(rightArr[c - 'a'] == 0)
                right--;

            if(right == left)
                count++;

        }
        return count;
    }

    public int minFallingPathSum(int[][] matrix ){

        if(matrix == null || matrix.length == 0|| matrix[0].length == 0)
            return 0;

        int m = matrix.length, n = matrix[0].length;

        int[][] M = new int[m][n];

        for(int i = 0; i < n; i++)
            M[0][i] = matrix[0][i];

        for(int i = 1; i < m; i++){
            for(int j = 0; j < n; j++){

                if(j == 0){
                    M[i][j] = Math.min(M[i - 1][j], M[i - 1][j + 1]);
                } else if(j == n - 1){
                    M[i][j] = Math.min(M[i - 1][j - 1], M[i - 1][j]);
                } else {
                    M[i][j] = Math.min(M[i - 1][j + 1], Math.min(M[i - 1][j], M[i - 1][j - 1]));
                }

                M[i][j] += matrix[i][j];
            }
        }

        int ans = Integer.MAX_VALUE;
        for(int num: M[m - 1])
            ans = Math.min(ans, num);

        return ans;
    }

    public static int mincostTickets(int[] days, int[] costs) {

//        int last = days[days.length - 1];
//        int[] dp = new int[last + 1];
//        boolean[] isTravel = new boolean[last + 1];
//
//        for(int day: days)
//            isTravel[day] = true;
//
//        for(int i = 1; i <= last; i++){
//            if(!isTravel[i]){
//                dp[i] = dp[i - 1];
//                continue;
//            }
//
//            dp[i] = dp[i - 1] + costs[0];
//            dp[i] = Math.min(costs[1] + dp[Math.max(i - 7, 0)], dp[i]);
//            dp[i] = Math.min(costs[2] + dp[Math.max(i - 30, 0)], dp[i]);
//        }
//        return dp[last];

        int last = days[days.length - 1];
        int[] dp = new int[last + 1];
        Set<Integer> set = new HashSet<>();

        for(int day: days)
            set.add(day);

        for(int i = 1; i <= last; i++){

            int c1 = dp[i- 1];
            int c2 = dp[Math.max(i - 7,  0)];
            int c3 = dp[Math.max(i - 30, 0)];

            if(!set.contains(i)){
                dp[i] = dp[i - 1];

            } else {
                dp[i] = Math.min(c1 + costs[0], Math.min(c2 + costs[1], c3 + costs[2]));
            }

        }
        return dp[last];
    }

    int ct = 0;
    public void helperArrangement(int n, int pos, int[] used){

        if(pos > n){
            ct++;
            return;
        }

        for(int i = 1; i<= n; i++){
            if(used[i] == 0 && (i % pos == 0 || pos % i == 0)){
                used[i] = 1;
                helperArrangement(n, pos + 1, used);
                used[i] = 0;
            }
        }
    }
    public int countArrangement(int n){
        if(n == 0)
            return 0;
        helperArrangement(n, 1, new int[n + 1]);
        return ct;
    }

    public int waysToMakeFair(int[] nums) {

        int ans = 0, n = nums.length;
        int rightEven = 0, rightOdd = 0;

        for(int i = 0; i < n; i++){
            if(i % 2 == 0)
                rightEven += nums[i];
            else
                rightOdd += nums[i];
        }

        int leftEven = 0, leftOdd = 0;

        for(int i = 0; i < n; i++){

            if(i % 2 == 0)
                rightEven -= nums[i];
            else
                rightOdd -= nums[i];

            if(leftEven + rightOdd == rightEven + leftOdd)
                ans++;

            if(i % 2 == 0)
                leftEven += nums[i];
            else
                leftOdd += nums[i];
        }
        return ans;
    }

    public int minCost(String s, int[] cost){

        int n = s.length(), ans = 0;
        for(int i = 1; i < n; i++){

            if(s.charAt(i) == s.charAt(i - 1)){

                ans = ans + Math.min(cost[i], cost[i - 1]);

                cost[i] = Math.max(cost[i], cost[i - 1]);
            }
        }
        return ans;
    }

    public int longestCommonSubsequence(String text1, String text2) {


        int m = text1.length(), n = text2.length();
        if(m == 0 || n == 0)
            return 0;
        int[][] dp = new int[m + 1][n + 1];

        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                if(text1.charAt(i - 1) == text2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else{
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[m][n];

    }

    public int maxProfit(int[] prices, int fee) {

//        int n = prices.length, mn = prices[0], ans = 0;
//
//       for(int i = 0; i < n; i++){
//
//           if(prices[i] < mn){
//               mn = prices[i];
//           }
//           if(prices[i] > mn + fee){
//               ans += prices[i] - mn - fee;
//               mn = prices[i] - fee;
//           }
//       }
//        return ans;

        int cash = 0, hold = -prices[0] - fee;

        for(int i = 1; i < prices.length; i++){

            cash = Math.max(cash, hold + prices[i]);
            hold = Math.max(hold, cash - prices[i] - fee);
        }

        return cash;
    }

    public static int minSetSize(int[] arr) {

        Map<Integer, Integer> freq = new HashMap<>();
        for(int a: arr){
            freq.put(a, freq.getOrDefault(a, 0) + 1);
        }

        int[] frequencies = new int[freq.values().size()];
        int i = 0;
        for(int cnt: freq.values()){
            frequencies[i++] = cnt;
        }
        Arrays.sort(frequencies);

        int ans = 0, half = arr.length/2, removed = 0;
        i = frequencies.length - 1;

        while(removed < half){
            ans++;
            removed += frequencies[i--];
        }

        return ans;
    }

    Map<String, List<Integer>> map1 = new HashMap<>();

    private int calculate(int num1, char ch, int num2){
        switch (ch){
            case '+': return num1 + num2;

            case '-': return num1 - num2;

            case '*': return num1 * num2;
        }
        return -1;
    }
    private boolean isOperation(char ch){
        return ch == '+' || ch == '-' || ch == '*';
    }
    public List<Integer> diffWaysToCompute(String expression){
        if(expression.length() == 0){
            return new ArrayList<>();
        }

        if(map1.containsKey(expression))
            return map1.get(expression);

        List<Integer> ans = new ArrayList<>();
        int idx = 0, num = 0;
        while(idx < expression.length() && !isOperation(expression.charAt(idx))){
            num = num * 10 + expression.charAt(idx) - '0';
            idx++;
        }

        if(idx == expression.length()){
            ans.add(num);
            map1.put(expression, ans);
            return ans;
        }

        for(int i = 0; i < expression.length(); i++){
            if(isOperation(expression.charAt(i))){
                List<Integer> p1 = diffWaysToCompute(expression.substring(0, i));
                List<Integer> p2 = diffWaysToCompute(expression.substring(i + 1));

                for(int j = 0; j < p1.size(); j++){
                    for(int k = 0; k < p2.size(); k++){
                        char op = expression.charAt(i);

                        ans.add(calculate(p1.get(j), op, p2.get(k)));
                    }
                }
            }
        }
        map1.put(expression, ans);
        return ans;
    }
//    public List<Integer> diffWaysToCompute(String expression) {
//
//        List<Integer> ans = new ArrayList<>();
//        int len = expression.length();
//        for(int i = 0; i < len; i++){
//            char ch = expression.charAt(i);
//
//            if(ch == '+' || ch == '-' || ch == '*'){
//
//                String p1 = expression.substring(0, i);
//                String p2 = expression.substring(i + 1);
//                List<Integer> l1 = map1.getOrDefault(p1, diffWaysToCompute(p1));
//                List<Integer> l2 = map1.getOrDefault(p2, diffWaysToCompute(p2));
//
//                for(int i1: l1){
//                    for(int i2: l2){
//
//                        int temp = 0;
//                        switch (ch){
//                            case '+': {
//                                temp = i1 + i2;
//                                break;
//                            }
//                            case '-': {
//                                temp = i1 - i2;
//                                break;
//                            }
//                            case '*' : {
//                                temp = i1 * i2;
//                                break;
//                            }
//
//                        }
//
//                        ans.add(temp);
//                    }
//                }
//            }
//        }
//
//        if(ans.size() == 0){
//            ans.add(Integer.valueOf(expression));
//        }
//        map1.put(expression, ans);
//        return ans;
//    }

    public int longestSubarray(int[] nums) {

//        int i = 0, j = 0, k = 1, ans = 0;
//        for(j = 0; j < nums.length; j++){
//
//            if(nums[j] == 0)
//                k--;
//
//            while(k < 0){
//                if(nums[i] == 0)
//                    k++;
//                i++;
//            }
//
//
//            ans = Math.max(ans, j - i);
//        }
//
//        return ans;

        int idx = 0, zeros = 1, ans = 0;
        for(int i = 0; i < nums.length; i++){

            if(nums[i] == 0)
                zeros--;

            while(zeros < 0){
                if(nums[idx] == 0)
                    zeros++;
                idx++;
            }
            ans = Math.max(ans, i - idx);
        }
        return ans;
    }

    public static int longestOnes(int[] nums, int k) {

        int idx = 0, i, ans = 0;
        for(i = 0; i < nums.length; i++){

            if(nums[i] == 0)
                k--;
            while(k < 0){
                if(nums[idx] == 0)
                    k++;
                idx++;
            }
            ans = Math.max(ans, i - idx + 1);
        }
        return ans;
    }

    public int minHeightShelves(int[][] books, int shelf_width) {

        int[] dp = new int[books.length + 1];

        dp[0] = 0;

        for(int i = 1; i <= books.length; i++){
            int thickness = books[i - 1][0];
            int height = books[i - 1][1];

            dp[i] = dp[i - 1] + height;

            for(int j = i - 1; j > 0 && thickness + books[j - 1][0] <= shelf_width; j--){

                height = Math.max(height, books[j - 1][1]);
                thickness += books[j - 1][0];
                dp[i] = Math.min(dp[i], dp[j - 1] + height);
            }
        }
        return dp[books.length];
    }

    class Mat {
        int row;
        int col;

        Mat(int row, int col){
            this.row = row;
            this.col = col;
        }
    }
    public int kthSmallest(int[][] matrix, int k) {

//        PriorityQueue<Mat> pq = new PriorityQueue<>((a, b) -> matrix[a.row][a.col] - matrix[b.row][b.col]);
//
//        for(int i = 0; i < matrix.length && i < k; i++)
//            pq.add(new Mat(i, 0));
//
//        int idx = 0, ans = 0;
//        while(!pq.isEmpty()){
//            Mat mat = pq.poll();
//
//            ans = matrix[mat.row][mat.col];
//
//            if(++idx == k)
//                break;
//            mat.col++;
//
//            if(matrix[0].length > mat.col)
//                pq.add(mat);
//        }
//
//        return ans;

        int n = matrix.length, left = matrix[0][0], right = matrix[n - 1][n - 1];

        while(left < right){

            int mid = left + (right - left)/2;

            int[] pair = { matrix[0][0], matrix[n - 1][n - 1]};

            int count = countEqual(matrix, mid, pair);

            if(count == k)
                return pair[0];

            if(count < k)
                left = pair[1];
            else
                right = pair[0];
        }
        return left;
    }

    private static int countEqual(int[][] matrix, int mid, int[] pair){
        int count = 0, n = matrix.length, row = n - 1, col = 0;

        while(row >= 0 && col < n){

            if(matrix[row][col] > mid){

                pair[1] = Math.min(pair[1], matrix[row][col]);
                row--;
            } else {

                pair[0] = Math.max(pair[0], matrix[row][col]);
                count += row + 1;
                col++;
            }
        }

        return count;
    }

    public int lengthOfLIS(int[] nums){

        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);

        for(int i = 1; i < nums.length; i++){
            for(int j = 0; j < i; j++){

                if(nums[j] < nums[i]){
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        int ans = 0;
        for(int i: dp)
            ans = Math.max(ans, i);

        return ans;
    }

    public String frequencySort(String s) {

        if(s == null)
            return null;

        Map<Character, Integer> map = new HashMap<>();

        for(char c: s.toCharArray()){
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        PriorityQueue<Character> pq = new PriorityQueue<>((a, b) -> map.get(b) - map.get(a));

        for(Character c: map.keySet()){
            pq.offer(c);
        }

        StringBuilder sb = new StringBuilder();
        while(!pq.isEmpty()){
            char c = pq.poll();
            for(int i = 0; i < map.get(c); i++){
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private int dis(int[] point){
        return point[0] * point[0] + point[1] * point[1];
    }

    public int[][] kClosest(int[][] points, int k) {

        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> {
            return dis(points[b])- dis(points[a]);
        });

        for(int i = 0; i < points.length; i++){
            pq.add(i);

            if(pq.size() > k)
                pq.remove();

        }

        int[][] ans = new int[k][];

        for(int i = 0; i < k; i++){
            int idx = pq.remove();
            ans[i] = points[idx];
        }

        return ans;
    }

    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {

        int rows = obstacleGrid.length, cols = obstacleGrid[0].length;

        if(obstacleGrid[0][0] == 1)
            return 0;

        obstacleGrid[0][0] = 1;

        for(int i = 1; i < rows; i++){
            obstacleGrid[i][0] = (obstacleGrid[i][0] == 0 && obstacleGrid[i - 1][0] == 1 ) ? 1: 0;
        }

        for(int i = 1; i< cols; i++){
            obstacleGrid[0][i] = (obstacleGrid[0][i] == 0 && obstacleGrid[0][i - 1] == 1) ? 1 : 0;
        }

        for(int i = 1; i < rows; i++){
            for(int j = 1; j < cols; j++){
                if(obstacleGrid[i][j] == 0){
                    obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
                } else {
                    obstacleGrid[i][j] = 0;
                }
            }
        }

        return obstacleGrid[rows - 1][cols - 1];
    }

    public int minPathSum(int[][] grid) {

        int rows = grid.length, cols = grid[0].length;

        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){

                if(i == 0 && j == 0){
                    grid[i][j] = grid[i][j];
                } else if(i == 0 && j != 0){
                    grid[i][j] = grid[i][j] + grid[i][j- 1];
                } else if(i != 0 && j == 0){
                    grid[i][j] = grid[i][j] + grid[i - 1][j];
                } else {
                    grid[i][j] = grid[i][j] + Math.min(grid[i - 1][j], grid[i][j - 1]);
                }
            }
        }

        return grid[rows - 1][cols - 1];
    }

    public int numTrees(int n) {

        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;

        for(int i = 2; i <= n; i++ ){
            for(int j = 1; j <= i; j++){
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }

        return dp[n];
    }

    public int minimumTotal(List<List<Integer>> triangle) {

        int n = triangle.size();
        int[] dp = new int[n];

        //last row
        for(int i = 0; i < n; i++){
            dp[i] = triangle.get(n - 1).get(i);
        }

        //bottom - up
        for(int i = n - 2; i >= 0; i--){
            for(int j = 0; j <= i; j++){
                dp[j] = Math.min(dp[j], dp[j +1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }

    public boolean wordBreak(String s, List<String> wordDict) {

        Set<String> set = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        for(int i = 1; i <= n; i++){
            for(int j = 0; j < i; j++){

                if(dp[j] && set.contains(s.substring(j, i))){
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n];
    }

    public int coinChange(int[] coins, int amount) {

        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);

        dp[0] = 0;
        for(int i = 1; i <= amount; i++){

            for(int coin: coins){
                if(i - coin < 0)
                    continue;
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }

        return dp[amount] == (amount + 1) ? -1: dp[amount];
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){

        if(root == null)
            return null;

        if(root.val == p.val || root.val == q.val)
            return root;

        TreeNode lLca = lowestCommonAncestor(root.left, p, q);
        TreeNode rLca = lowestCommonAncestor(root.right, p, q);

        if(lLca != null && rLca != null)
            return root;

        return (lLca != null) ? lLca : rLca;

    }


    public String customSortString(String order, String str) {

        int[] count = new int[26];

        for(char ch: str.toCharArray()){
            ++count[ch - 'a'];
        }

        StringBuilder sb = new StringBuilder();

        for(char ch: order.toCharArray()){
            while(count[ch - 'a']-- > 0){
                sb.append(ch);
            }
        }

        for(char ch = 'a'; ch <= 'z'; ch++){
            while(count[ch - 'a']-- > 0){
                sb.append(ch);
            }
        }

        return sb.toString();
    }

    public int deleteAndEarn(int[] nums) {

        int[] count = new int[10002];
        for(int i: nums){
            count[i]++;
        }

        int[] dp = new int[10002];
        dp[0] = 0;
        dp[1] = count[1];

        for(int i = 2; i < count.length; i++){
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + count[i] * i);
        }
        return dp[10001];


    }

    public int maximalSquare(char[][] matrix) {

        int rows = matrix.length, cols = rows > 0 ? matrix[0].length : 0;

        int[][] dp = new int[rows + 1][cols + 1];
        int max = 0;

        for(int i = 1; i <= rows; i++){
            for(int j = 1; j <= cols; j++){

                if(matrix[i - 1][j - 1] == '1'){
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    max = Math.max(max, dp[i][j]);
                }
            }
        }

        return max * max;

    }

    public int maxProduct(int[] nums) {

        int n = nums.length, ans = Integer.MIN_VALUE;

        int max[] = new int[n];
        int min[] = new int[n];

        max[0] = nums[0];
        min[0] = nums[0];

        for(int i = 1; i < n; i++){
            max[i] = Math.max(max[i - 1] * nums[i], Math.max(min[i - 1] * nums[i], nums[i]));
            min[i] = Math.min(max[i - 1] * nums[i], Math.min(min[i - 1] * nums[i], nums[i]));
        }

        for(int i: max){
            ans = Math.max(ans, i);
        }

        return ans;
    }

    public int integerBreak(int n) {

        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for(int i = 2; i <= n; i++){
            for(int j = 1; j < i; j++){

                dp[i] = Math.max(dp[i], Math.max(j * dp[j], (i - j) * dp[i - j]));
            }
        }

        return dp[n];
    }

    public int goodNodesfunc(TreeNode node, int max){
        if(node == null)
            return 0;

        int temp = (node.val >= max) ? 1 : 0;

        temp += goodNodesfunc(node.left, Math.max(max, node.val));
        temp += goodNodesfunc(node.right, Math.max(max, node.val));

        return temp;
    }
    public int goodNodes(TreeNode root) {

        return goodNodesfunc(root, -100001);
    }

    class Node {
        public int val;
        public Node left;
        public Node right;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val,Node _left,Node _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    };

    //426. Convert Binary Search Tree to Sorted Doubly Linked List
    Node first = null, last = null;
    public void treeToDoublyListFunc(Node node){
        if(node != null){

            treeToDoublyListFunc(node.left);

            if(last != null){

                last.right = node;
                node.left = last;

            } else {
                first = node;
            }
            last = node;

            treeToDoublyListFunc(node.right);
        }
    }
    public Node treeToDoublyList(Node root) {

        if(root == null)
            return null;

        treeToDoublyListFunc(root);
        last.right = first;
        first.left = last;

        return first;

    }

    public int triangleNumber(int[] nums){
        int count = 0;
        Arrays.sort(nums);

        for(int i = 0; i < nums.length - 2; i++){
            int k = i + 2;
            for(int j = i + 1; j < nums.length - 1 && nums[i] != 0; j++){

                while(k < nums.length && nums[i] + nums[j] > nums[k]){

                    k++;
                    count += k -j - 1;
                }
            }
        }

        return count;
    }

    public int change(int amount, int[] coins) {

        int[] dp = new int[amount + 1];
        dp[0] = 1;

        for(int coin: coins){
            for(int x = coin; x < amount + 1; x++){
                dp[x] = dp[x] + dp[x -coin];
            }
        }

        return dp[amount];
    }

    public int rob(int[] nums) {

        int n = nums.length;

        if(n == 0)
            return 0;

        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = nums[0];

        for(int i = 2; i <= n; i++){
            dp[i] = Math.max(dp[i - 1] , dp[i - 2] + nums[i - 1]);
        }

        return dp[n];
    }

    public int countSubstrings(String s) {

        //Basic DP approach
//        int n = s.length(), ans = 0;
//
//        if(n <= 0)
//            return 0;
//        boolean[][] dp = new boolean[n][n];
//
//        //one character substrings
//        for(int i = 0; i < n; i++){
//            dp[i][i] = true;
//            ans++;
//        }
//
//        //two characters substrings
//        for(int i = 0; i < n - 1; i++){
//            dp[i][i + 1] = (s.charAt(i) == s.charAt(i + 1));
//
//            ans += dp[i][i + 1] ? 1 : 0;
//        }
//
//        //more than 3 characters
//        for(int len = 3; len <= n; len++){
//            for(int i = 0, j = i + len - 1; j < n; i++, j++){
//                dp[i][j] = dp[i + 1][j -  1] && (s.charAt(i)== s.charAt(j));
//
//                ans += dp[i][j] ? 1 : 0;
//
//            }
//        }
//
//        return ans;

        //Based on longest palindromic subsequence
        int n = s.length(), ans = 0;
        boolean[][] dp = new boolean[n][n];

        for(int i = n - 1; i >= 0; i--){
            for(int j = i ; j < n; j++){

                dp[i][j] = (s.charAt(i) == s.charAt(j)) && (j - i < 3 || dp[i + 1][j - 1]);

                if(dp[i][j])
                    ans++;
            }
        }

        return ans;
    }

    public int findKthLargest(int[] nums, int k) {

        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> a - b);

        for(int i = 0; i < nums.length; i++){
            pq.add(nums[i]);
            if(pq.size() > k)
                pq.poll();
        }

        return pq.peek();
    }

    public List<String> topKFrequent(String[] words, int k) {

        List<String> ans = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();

       for(int i = 0; i < words.length; i++){
           map.put(words[i], map.getOrDefault(words[i], 0) + 1);
       }

       PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>((a, b) -> a.getValue() == b.getValue() ? b.getKey().compareTo(a.getKey()) : a.getValue() - b.getValue());

       for(Map.Entry<String, Integer> entry: map.entrySet()){
           pq.offer(entry);
           if(pq.size() > k)
               pq.poll();
       }
       while(!pq.isEmpty()){
           ans.add(0, pq.poll().getKey());
       }

       return ans;
    }

    public int leastInterval(char[] tasks, int n) {

        int[] freq = new int[26];
        for(int t: tasks)
            freq[t - 'A']++;

        Arrays.sort(freq);

        int ch_max = freq[25];
        int idle_time = (ch_max - 1) * n;

        for(int i = freq.length - 2; i >= 0 && idle_time > 0; i--){
            idle_time -= Math.min(ch_max - 1, freq[i]);
        }
        idle_time = Math.max(0, idle_time);

        return idle_time + tasks.length;
    }

    public int[][] merge(int[][] intervals) {

        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));

        LinkedList<int[]> ans = new LinkedList<>();
        for(int[] interval : intervals){
            if(ans.isEmpty() || ans.getLast()[1] < interval[0]){
                ans.add(interval);
            } else {
                ans.getLast()[1] = Math.max(ans.getLast()[1], interval[1]);
            }
        }

        return ans.toArray(new int[ans.size()][]);
    }

    public void dfsItinerary(String departure, Map<String, PriorityQueue<String>> map, LinkedList<String> ans){

        PriorityQueue<String> arrivals = map.get(departure);

        while(arrivals != null && !arrivals.isEmpty()){
                dfsItinerary(arrivals.poll(), map, ans);
        }

        ans.addFirst(departure);
    }
    public List<String> findItinerary2(List<List<String>> tickets, Map<String, PriorityQueue<String>> map, LinkedList<String> ans){

        for(List<String> ticket: tickets){
            map.putIfAbsent(ticket.get(0), new PriorityQueue<>());
            map.get(ticket.get(0)).add(ticket.get(1));
        }

        dfsItinerary("JFK", map, ans);
        return ans;
    }
    public List<String> findItinerary(List<List<String>> tickets) {
        Map<String, PriorityQueue<String>> map = new HashMap<>();
        LinkedList<String> ans = new LinkedList<>();

        return findItinerary2(tickets, map, ans);
    }

    public void dfsFindCircleNum(int[][] isConnected, boolean[] visited, int i){
        visited[i] = true;
        for(int j = 0; j < isConnected.length; j++){
            if(isConnected[i][j] == 1 && !visited[j])
                dfsFindCircleNum(isConnected, visited, j);
        }
    }
    public int findCircleNum(int[][] isConnected) {
        boolean[] visited = new boolean[isConnected.length];

        int count = 0;
        for(int i = 0; i < isConnected.length; i++){
            if(!visited[i] ){
                dfsFindCircleNum(isConnected, visited, i);
                count++;
            }
        }
//        Queue<Integer> q = new LinkedList<>();
//        for(int i = 0; i < isConnected.length; i++){
//            if(!visited[i]){
//                q.add(i);
//                while(!q.isEmpty()){
//                    int temp = q.poll();
//                    visited[temp] = true;
//                    for(int j = 0; j < isConnected.length; j++){
//                        if(isConnected[temp][j] == 1 && !visited[j]){
//                            q.add(j);
//                        }
//                    }
//                }
//                count++;
//            }
//
//        }

        return count;
    }

    //323 Number of Connected Components in Undirected Graph
    public void dfsCountComponents(List<Integer>[] adj_lst, int[] visited, int i){
        visited[i] = 1;

        for(int j = 0; j < adj_lst[i].size(); j++){
            if(visited[adj_lst[i].get(j)] == 0 ){
                dfsCountComponents(adj_lst, visited, adj_lst[i].get(j));
            }
        }
    }
    public int countComponents(int n, int[][] edges){

        int components = 0;
        int[] visited = new int[n];

        List<Integer>[] adj_lst = new ArrayList[n];
        for(int i = 0; i < n; i++){
            adj_lst[i] = new ArrayList<>();
        }

        for(int i = 0; i < edges.length; i++){
            adj_lst[edges[i][0]].add(edges[i][1]);
            adj_lst[edges[i][1]].add(edges[i][0]);
        }

        for(int i = 0; i < n; i++){
            if(visited[i] == 0){
                dfsCountComponents(adj_lst, visited, i);
                components++;
            }
        }

        return components;
    }

    public boolean isCyclic(ArrayList<Integer>[] graph, int[] visited, int i){
        if(visited[i] == 2)
            return true;

        visited[i] = 2;

        for(int j = 0; j < graph[i].size(); j++){
            if(visited[graph[i].get(j)] != 1){
                if(isCyclic(graph, visited, graph[i].get(j)))
                    return true;
            }
        }
        visited[i] = 1;
        return false;
    }
    public boolean canFinish(int numCourses, int[][] prerequisites) {

        ArrayList<Integer>[] graph = new ArrayList[numCourses];
        for(int i = 0; i < numCourses; i++){
            graph[i] = new ArrayList();
        }

        int[] visited = new int[numCourses];
        Arrays.fill(visited, 0);

        for(int i = 0; i < prerequisites.length; i++){
            graph[prerequisites[i][1]].add(prerequisites[i][0]);
        }

        for(int i = 0; i < numCourses; i++){

            if(visited[i] == 0){
                if(isCyclic(graph, visited, i))
                    return false;
            }

        }
        return true;
    }

    public int longestStrChain(String[] words) {

        Map<String, Integer> dp = new HashMap<>();

        Arrays.sort(words, (a,b) -> a.length() - b.length());
        int longestWordSequenceLength = 1;

        for(String word: words){
            int presentLength = 1;

            for(int i = 0; i < word.length(); i++){
                StringBuilder temp = new StringBuilder(word);
                temp.deleteCharAt(i);
                String predecessor = temp.toString();
                int previousLength = dp.getOrDefault(predecessor, 0);

                presentLength = Math.max(presentLength, previousLength + 1);
            }
            dp.put(word, presentLength);
            longestWordSequenceLength = Math.max(longestWordSequenceLength, presentLength);
        }

        return longestWordSequenceLength;
    }

    public int findTargetSumWays(int[] nums, int target) {

//       int sum = 0;
//       for(int i: nums)
//           sum += i;
//       if(target > sum || target < -sum)
//           return 0;
//       int[] dp = new int[2 * sum + 1];
//
//       dp[0 + sum] = 1;
//       for(int i = 0; i < nums.length; i++){
//
//           int[] next = new int[2 * sum + 1];
//           for(int j = 0; j < 2 * sum + 1; j++){
//
//               if(dp[j] != 0){
//                   next[j + nums[i]] += dp[j];
//                   next[j - nums[i]] += dp[j];
//               }
//           }
//           dp = next;
//       }
//
//       return dp[sum + target];

        Map<Integer, Integer> dp = new HashMap<>();
        dp.put(0, 1);

        for(int num: nums){
            Map<Integer, Integer> dp2 = new HashMap<>();
            for(int tempSum: dp.keySet()){
                int key1 = tempSum + num;
                dp2.put(key1, dp2.getOrDefault(key1, 0) + dp.get(tempSum));
                int key2 = tempSum - num;
                dp2.put(key2, dp2.getOrDefault(key2, 0) + dp.get(tempSum));
            }
            dp = dp2;
        }

        return dp.getOrDefault(target, 0);
    }

    private boolean topologicalSort(List<List<Integer>> adj_lst, int v, Stack<Integer> stack, int[] visited){

        if(visited[v] == 2)
            return true;
        if(visited[v] == 1)
            return false;

        visited[v] = 1;

        for(Integer i: adj_lst.get(v)){
            if(!topologicalSort(adj_lst, i, stack, visited))
                return false;
        }

        visited[v] = 2;
        stack.push(v);

        return true;
    }
    public int[] findOrder(int numCourses, int[][] prerequisites) {

        List<List<Integer>> adj_lst = new ArrayList<>();
        for(int i = 0; i < numCourses; i++)
            adj_lst.add(new ArrayList<>());

        for(int i = 0; i < prerequisites.length; i++){
            adj_lst.get(prerequisites[i][1]).add(prerequisites[i][0]);
        }
        Stack<Integer> stack = new Stack<>();

        int[] visited = new int[numCourses];
        for(int i = 0; i < numCourses; i++){
            if(!topologicalSort(adj_lst, i, stack, visited)){
                return new int[0];
            }
        }
        int i = 0;
        int[] ans = new int[numCourses];
        while(!stack.isEmpty()){
            ans[i++] = stack.pop();
        }
        return ans;
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {

        Map<String, Set<String>> graph = new HashMap<>();
        Map<String, String> map = new HashMap<>();

        //build the graph
        for(List<String> account : accounts){

            String userName = account.get(0);
            for(int i = 1; i < account.size(); i++){

                if(!graph.containsKey(account.get(i))){
                    graph.put(account.get(i), new HashSet<>());
                }

                map.put(account.get(i), userName);

                if(i == 1)
                    continue;
                graph.get(account.get(i)).add(account.get(i - 1));
                graph.get(account.get(i - 1)).add(account.get(i));
            }
        }

        Set<String> visited = new HashSet<>();
        List<List<String>> ans = new LinkedList<>();

        //DFS
        for(String email: map.keySet()){
            List<String> list = new LinkedList<>();

            if(visited.add(email)){
                dfsAccountsMerge(graph, email, visited, list);
                Collections.sort(list);
                list.add(0, map.get(email));
                ans.add(list);
            }
        }

        return ans;

    }

    public void dfsAccountsMerge(Map<String, Set<String>> graph, String email, Set<String> visited, List<String> list){
        list.add(email);
        for(String next: graph.get(email)){
            if(visited.add(next)){
                dfsAccountsMerge(graph, next, visited, list);
            }
        }

    }

    public int partitionDisjoint(int[] nums) {

        int len = 1;
        int arrMax = nums[0];
        int secMax = 0;

        for(int i = 0; i < nums.length; i++){
            secMax = Math.max(nums[i], secMax);

            if(nums[i] < arrMax){
                arrMax = secMax;
                len = i + 1;
            }
        }

        return len;
    }

//    //Uniques BST II
//    public List<TreeNode> generateTrees(int n) {
//
//    }


    public static void main(String[] args) {

        //System.out.println(convert("PAYPALISHIRING", 3));
        //System.out.println(myAtoi(" "));
        //System.out.println(isNStraightHand(new int[]{1,2,3,6,2,3,4,7,8}, 3));
        //System.out.println(pushDominoes(".L.R...LR..L.."));
//        int cand[] = new int[]{2,3,6,7};
//        int target = 7;
//        System.out.println(new Medium().combinationSum(cand, target));
//            int[][] matrix = new int[][]{{1,2,3}, {4,5,6}, {7,8,9}};
//            new Medium().rotate(matrix);
//            System.out.println(matrix);
        //System.out.println(mincostTickets(new int[]{1, 4, 6, 7, 8, 20}, new int[]{2, 7, 15}));
//        System.out.println(minSetSize(new int[]{3,3,3,3,5,5,5,2,2,7}));
//        System.out.println(new Medium().diffWaysToCompute("2-1-1"));
        //System.out.println(longestOnes(new int[]{1,1,1,0,0,0,1,1,1,1,0}, 2));
        //System.out.println(new Medium().kthSmallest(new int[][]{{1,5,9}, {10, 11, 13}, {12, 13, 15}}, 8));
        //System.out.println(uniquePathsWithObstacles(new int[][]{{0,0,0}, {0, 1, 0}, {0,0,0}}));
       // System.out.println(new Medium().coinChange(new int[]{1,2,5}, 5));
        //System.out.println(new Medium().deleteAndEarn(new int[]{2,2,3,3,3,4}));
        //System.out.println(new Medium().triangleNumber(new int[]{4,2,3,4}));
        //System.out.println(new Medium().change(5, new int[]{1,2,5}));
        //System.out.println(new Medium().rob(new int[]{1,2,3,1}));
        //System.out.println(new Medium().leastInterval(new char[]{'A', 'A', 'A', 'B', 'B', 'B'}, 2));
       // System.out.println(new Medium().longestStrChain(new String[]{"a","b","ba","bca","bda","bdca"}));
        //System.out.println(new Medium().findTargetSumWays(new int[]{1,1,1,1,1}, 3));

    }
}
