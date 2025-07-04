import { message } from 'antd';
import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { useNavigate, useParams } from "react-router-dom";
import { BarLoader } from 'react-spinners';
import Styles from "./quiz.module.css";

const Quiz = () => {
    const [loading, setLoading] = useState(false);
    const { topicId } = useParams();
    const [quizData, setQuizData] = useState([]);
    const [quizId, setQuizId] = useState(null);
    const [response, setResponse] = useState(null);
    const [selectedOptions, setSelectedOptions] = useState([]);
    const [descriptiveAnswers, setDescriptiveAnswers] = useState([]);
    const [mcqAnswers, setMcqAnswers] = useState([]);
    const [descAnswers, setDescAnswers] = useState([]);
    const [score, setScore] = useState(null);
    const navigate = useNavigate();

    const getQuiz = async () => {
        try {
            setLoading(true);
            const res = await fetch(`http://localhost:3001/users/${topicId}/getquiz`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
            });
            if (res.status === 404) {
                message.info("Let's Generate Quiz");
                return;
            }
            const fetchedData = await res.json();
            setQuizData(fetchedData.quizArray);
            setQuizId(fetchedData._id);
            setSelectedOptions(Array(fetchedData.quizArray.mcq.length).fill(null));
            setDescriptiveAnswers(Array(fetchedData.quizArray.descriptive.length).fill(""));
        } catch (error) {
            console.error('Data not found', error.message);
        }
        finally {
            setLoading(false);
        }
    };
    const getnewQuiz = async () => {
        try {
            setLoading(true);
            const res = await fetch(`http://localhost:3001/users/${topicId}/addorupdatequiz`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            if (res.status === 404) {
                message.info("Let's Generate Quiz");
                return;
            }
            const fetchedData = await res.json();
            setQuizData(fetchedData.quiz.quizArray);
            setQuizId(fetchedData._id);
            setSelectedOptions(Array(fetchedData.quiz.quizArray.mcq.length).fill(null));
            setDescriptiveAnswers(Array(fetchedData.quiz.quizArray.descriptive.length).fill(""));
        } catch (error) {
            console.error('Data not found', error.message);
        }
        finally {
            setLoading(false);
        }
    };
    // Generate a new quiz
    const generateNewQuiz = async () => {
        await getnewQuiz();
    };
    const user_id = useSelector((state) => state.user?._id);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const hasNullOption = selectedOptions.some(option => option === null);
        const hasNulldesc = descriptiveAnswers.some(option => option === '');
        if (hasNullOption || hasNulldesc) {
            message.info("Complete all questions");
            return;
        }
        try {
            setLoading(true);
            const res = await fetch(`http://localhost:3001/users/${topicId}/verify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id, selectedOptions, descriptiveAnswers })
            });
            const fetchedData = await res.json();
            setMcqAnswers(fetchedData.answer)
            setDescAnswers(fetchedData.answer1)
            setScore(fetchedData.scorePercentage)
            if (res.status === 200) {
                window.scrollTo({
                    top: 0,
                    behavior: "smooth"
                });
                message.success(fetchedData.message);
            } else {
                message.error('Some error occurred');
            }
        } catch (error) {
            console.log(error);
            message.error('Some error occurred');
        } finally {
            setLoading(false);
        }
    };

    const handleOptionChange = (questionIndex, optionValue) => {
        const newSelectedOptions = [...selectedOptions];
        newSelectedOptions[questionIndex] = optionValue;
        setSelectedOptions(newSelectedOptions);
    };

    const handleDescriptiveChange = (questionIndex, value) => {
        const newDescriptiveAnswers = [...descriptiveAnswers];
        newDescriptiveAnswers[questionIndex] = value;
        setDescriptiveAnswers(newDescriptiveAnswers);
    };
    const data = useSelector((state) => state.user);
    useEffect(() => {
        if (!data) {
            navigate(-1);
            message.error("login first")
            return;
        }
        getQuiz();
    }, []);

    return (
        <>
            {loading ? (
                <div className={Styles.loading}>
                    <BarLoader />
                </div>
            ) : (
                <div className={Styles.container}>
                    <div className={Styles.score}>
                        {score!==null && <p>You Scored - {score}%</p>}
                    </div>
                    {quizData && quizData?.mcq && quizData?.mcq.map((questionObj, index) => (
                        <div key={index} className={Styles.container_ques}>
                            <h3><strong>{index + 1}. </strong>{questionObj.question}</h3>
                            <ul className={Styles.container_opt}>
                                {Object.entries(questionObj.options).map(([key, value]) => (
                                    <li key={key}>
                                        <input
                                            type="radio"
                                            name={`question_${index}`}
                                            value={key}
                                            onChange={() => handleOptionChange(index, key)}
                                            checked={selectedOptions[index] === key}
                                        />
                                        &nbsp;
                                        <label>{value}</label>
                                    </li>
                                ))}
                            </ul>
                            {mcqAnswers.length > 0 &&
                                <div>
                                    <p>Your Answer:  <strong>{mcqAnswers[index]?.your_answer}</strong></p>
                                    <p>Correct Answer: <strong>{mcqAnswers[index]?.right_answer}</strong></p>
                                </div>
                            }
                        </div>
                    ))}

                    {quizData && quizData?.descriptive && quizData?.descriptive.map((questionObj, index) => (
                        <div key={index} className={Styles.container_ques}>
                            <h3><strong>{quizData?.mcq.length + index + 1}. </strong>{questionObj.question}</h3>
                            <textarea
                                name={`question_desc_${index}`}
                                value={descriptiveAnswers[index]}
                                onChange={(e) => handleDescriptiveChange(index, e.target.value)}
                                className={Styles.textarea}
                            />{mcqAnswers.length > 0 &&
                                <div>
                                    <p>Your Answer:  <strong>{descAnswers[index]?.right_answer}</strong></p>
                                    <p> Correct Answer:<strong>{descAnswers[index]?.your_answer}</strong></p>
                                    <p>Accuracy: <strong>{descAnswers[index]?.accuracy}</strong></p>
                                </div>}
                        </div>
                    ))}
                    <div className={Styles.cont_button_div}>

                        {(quizData?.mcq?.length > 0 || quizData?.descriptive?.length > 0) && (
                            <button onClick={handleSubmit} className={Styles.cont_button}>
                                Submit
                            </button>
                        )}
                        <button className={Styles.cont_button_new} onClick={generateNewQuiz}>
                            Generate New Quiz
                        </button>
                    </div>

                </div>
            )}
        </>
    );
};

export default Quiz;
