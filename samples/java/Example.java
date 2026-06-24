public class Example {
    private static final int MAX_RETRIES = 3;

    void helper() {
        System.out.println("helper");
    }

    int computeDiscountedTotalForCustomerOrder(int quantity) {
        int total = 0;
        for (int i = 0; i < quantity; i++) {
            if (i % 2 == 0 && i > 0) {
                total += i * 7;
            } else {
                total += 42;
            }
        }
        helper();
        return total;
    }

    void caller() {
        computeDiscountedTotalForCustomerOrder(10);
    }
}
