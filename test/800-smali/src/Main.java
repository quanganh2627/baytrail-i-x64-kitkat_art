/*
 * Copyright (C) 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.LinkedList;
import java.util.List;

/**
 * Smali excercise.
 */
public class Main {

    private static class TestCase {
        public TestCase(String testName, String testClass, String testMethodName, Object[] values,
                        Throwable expectedException, Object expectedReturn) {
            this.testName = testName;
            this.testClass = testClass;
            this.testMethodName = testMethodName;
            this.values = values;
            this.expectedException = expectedException;
            this.expectedReturn = expectedReturn;
        }

        String testName;
        String testClass;
        String testMethodName;
        Object[] values;
        Throwable expectedException;
        Object expectedReturn;
    }

    private List<TestCase> testCases;

    public Main() {
        // Create the test cases.
        testCases = new LinkedList<TestCase>();

        testCases.add(new TestCase("b/17790197", "B17790197", "getInt", null, null, 100));
        testCases.add(new TestCase("b/17978759", "B17978759", "test", null, new VerifyError(), null));
        testCases.add(new TestCase("FloatBadArgReg", "FloatBadArgReg", "getInt",
            new Object[]{100}, null, 100));
        testCases.add(new TestCase("negLong", "negLong", "negLong", null, null, 122142L));
    }

    public void runTests() {
        for (TestCase tc : testCases) {
            System.out.println(tc.testName);
            try {
                runTest(tc);
            } catch (Exception exc) {
                exc.printStackTrace(System.out);
            }
        }
    }

    private void runTest(TestCase tc) throws Exception {
        Exception errorReturn = null;
        try {
            Class<?> c = Class.forName(tc.testClass);

            Method[] methods = c.getDeclaredMethods();

            // For simplicity we assume that test methods are not overloaded. So searching by name
            // will give us the method we need to run.
            Method method = null;
            for (Method m : methods) {
                if (m.getName().equals(tc.testMethodName)) {
                    method = m;
                    break;
                }
            }

            if (method == null) {
                errorReturn = new IllegalArgumentException("Could not find test method " +
                                                           tc.testMethodName + " in class " +
                                                           tc.testClass + " for test " +
                                                           tc.testName);
            } else {
                Object retValue;
                if (Modifier.isStatic(method.getModifiers())) {
                    retValue = method.invoke(null, tc.values);
                } else {
                    retValue = method.invoke(method.getDeclaringClass().newInstance(), tc.values);
                }
                if (tc.expectedException != null) {
                    errorReturn = new IllegalStateException("Expected an exception in test " +
                                                            tc.testName);
                }
                if (tc.expectedReturn == null && retValue != null) {
                    errorReturn = new IllegalStateException("Expected a null result in test " +
                                                            tc.testName);
                } else if (tc.expectedReturn != null &&
                           (retValue == null || !tc.expectedReturn.equals(retValue))) {
                    errorReturn = new IllegalStateException("Expected return " +
                                                            tc.expectedReturn +
                                                            ", but got " + retValue);
                } else {
                    // Expected result, do nothing.
                }
            }
        } catch (Throwable exc) {
            if (tc.expectedException == null) {
                errorReturn = new IllegalStateException("Did not expect exception", exc);
            } else if (!tc.expectedException.getClass().equals(exc.getClass())) {
                errorReturn = new IllegalStateException("Expected " +
                                                        tc.expectedException.getClass().getName() +
                                                        ", but got " + exc.getClass(), exc);
            } else {
              // Expected exception, do nothing.
            }
        } finally {
            if (errorReturn != null) {
                throw errorReturn;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Main main = new Main();

        main.runTests();

        System.out.println("Done!");
    }
}
